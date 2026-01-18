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
def announce_self_waiting(self, expected_version):
    """
        Announce this worker is waiting (via num_workers_waiting counter) to join next
        rendezvous, but only if state and version match.
        """
    while True:
        cas_delay()
        active_version, state = self.get_rdzv_state()
        if state['status'] != 'final' or state['version'] != expected_version:
            raise EtcdRendezvousRetryImmediately()
        state['num_workers_waiting'] += 1
        try:
            active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value)
            return active_version
        except etcd.EtcdCompareFailed:
            log.info('Announce self as waiting CAS unsuccessful, retrying')