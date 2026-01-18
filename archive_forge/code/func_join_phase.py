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
def join_phase(self, expected_version):
    """
        We observed a rendezvous state in 'joinable' state, and attempt to join this
        particular version, and then wait for all other peers to join.
        """
    active_version, this_rank = self.join_rendezvous(expected_version)
    state = json.loads(active_version.value)
    log.info('Joined rendezvous version %s as rank %s. Full state: %s', state['version'], this_rank, state)
    if this_rank == self._num_min_workers - 1 and state['status'] == 'joinable':
        log.info('Rank %s is responsible for join last call.', this_rank)
        last_call_deadline = time.time() + self._last_call_timeout
        self.handle_join_last_call(expected_version, last_call_deadline)
        log.info('Rank %s finished join last call.', this_rank)
    log.info('Waiting for remaining peers.')
    active_version = self.wait_for_peers(expected_version)
    state = json.loads(active_version.value)
    assert state['version'] == expected_version, 'Logic error: failed to observe version mismatch'
    return self.confirm_phase(expected_version, this_rank)