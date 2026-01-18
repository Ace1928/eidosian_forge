import json
import logging
import sys
import threading
import time
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed.elastic.rendezvous import (
from .utils import parse_rendezvous_endpoint
from .etcd_store import EtcdStore, cas_delay
def init_phase(self):
    """
        Initially, the rendezvous state is expected to be one of:

        1. empty (non-existent) - in this case we try to create a new one.
        2. joinable - we try to join it.
        3. final - we announce ourselves as waiting, and go into monitoring mode

        Any other state is considered transitional, and will be retried after
        a short delay.

        Returns:
            ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousClosedError - current rendezvous was/is closed
            EtcdRendezvousRetryableFailure - observed some intermediate
             state, which is best handled by retrying later
        """
    try:
        active_version = self.try_create_rendezvous()
        state = json.loads(active_version.value)
        log.info('New rendezvous state created: %s', state)
    except etcd.EtcdAlreadyExist:
        active_version, state = self.get_rdzv_state()
        log.info('Observed existing rendezvous state: %s', state)
    if state['status'] == 'closed':
        raise RendezvousClosedError()
    if state['status'] == 'joinable':
        return self.join_phase(state['version'])
    if state['status'] == 'final':
        self.handle_existing_rendezvous(state['version'])
        raise EtcdRendezvousRetryImmediately()
    self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
    raise EtcdRendezvousRetryableFailure()