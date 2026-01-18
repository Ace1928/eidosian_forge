from __future__ import absolute_import
import copy
import logging
import random
import threading
import time
import typing
from typing import Dict, Iterable, Optional, Union
from google.cloud.pubsub_v1.subscriber._protocol.dispatcher import _MAX_BATCH_LATENCY
from google.cloud.pubsub_v1.subscriber._protocol import requests
def maintain_leases(self) -> None:
    """Maintain all of the leases being managed.

        This method modifies the ack deadline for all of the managed
        ack IDs, then waits for most of that time (but with jitter), and
        repeats.
        """
    while not self._stop_event.is_set():
        deadline = self._manager._obtain_ack_deadline(maybe_update=True)
        _LOGGER.debug('The current deadline value is %d seconds.', deadline)
        leased_messages = copy.copy(self._leased_messages)
        cutoff = time.time() - self._manager.flow_control.max_lease_duration
        to_drop = [requests.DropRequest(ack_id, item.size, item.ordering_key) for ack_id, item in leased_messages.items() if item.sent_time < cutoff]
        if to_drop:
            _LOGGER.warning('Dropping %s items because they were leased too long.', len(to_drop))
            assert self._manager.dispatcher is not None
            self._manager.dispatcher.drop(to_drop)
        for item in to_drop:
            leased_messages.pop(item.ack_id)
        ack_ids = leased_messages.keys()
        expired_ack_ids = set()
        if ack_ids:
            _LOGGER.debug('Renewing lease for %d ack IDs.', len(ack_ids))
            assert self._manager.dispatcher is not None
            ack_id_gen = (ack_id for ack_id in ack_ids)
            expired_ack_ids = self._manager._send_lease_modacks(ack_id_gen, deadline)
        start_time = time.time()
        if self._manager._exactly_once_delivery_enabled() and len(expired_ack_ids):
            assert self._manager.dispatcher is not None
            self._manager.dispatcher.drop([requests.DropRequest(ack_id, leased_messages.get(ack_id).size, leased_messages.get(ack_id).ordering_key) for ack_id in expired_ack_ids if ack_id in leased_messages])
        snooze = random.uniform(_MAX_BATCH_LATENCY, deadline * 0.9 - (time.time() - start_time))
        _LOGGER.debug('Snoozing lease management for %f seconds.', snooze)
        self._stop_event.wait(timeout=snooze)
    _LOGGER.debug('%s exiting.', _LEASE_WORKER_NAME)