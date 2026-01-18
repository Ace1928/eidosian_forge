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
def confirm_membership(self, expected_version, this_rank):
    """Helper method for the confirm phase."""
    while True:
        cas_delay()
        active_version, state = self.get_rdzv_state()
        if state['status'] != 'frozen':
            raise EtcdRendezvousRetryImmediately('Rendezvous no longer frozen, before we confirmed. Must join next one')
        if state['version'] != expected_version:
            raise EtcdRendezvousRetryImmediately('Rendezvous version changed. Must try join the new one.')
        this_lease_key = self.get_path(f'/rdzv/v_{expected_version}/rank_{this_rank}')
        self.client.set(this_lease_key, value=None, ttl=CONST_WORKER_KEEPALIVE_TTL)
        state['keep_alives'].append(this_lease_key)
        if len(state['keep_alives']) == len(state['participants']):
            state['status'] = 'final'
            state['num_workers_waiting'] = 0
            finalize = True
        else:
            finalize = False
        try:
            active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=None if finalize else CONST_ETCD_FROZEN_TTL)
            self._lease_this_rank_stop = self.setup_lease_renewal(this_lease_key, ttl=CONST_WORKER_KEEPALIVE_TTL)
            return active_version
        except etcd.EtcdCompareFailed:
            log.info('Confirm membership CAS unsuccessful, retrying')