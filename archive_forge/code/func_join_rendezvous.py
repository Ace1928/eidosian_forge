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
def join_rendezvous(self, expected_version):
    """Helper method for the join phase."""
    while True:
        cas_delay()
        active_version, state = self.get_rdzv_state()
        if state['status'] != 'joinable':
            raise EtcdRendezvousRetryableFailure('Rendezvous state became non-joinable before we could join. Must join next one.')
        if state['version'] != expected_version:
            raise EtcdRendezvousRetryImmediately('Rendezvous version changed. Must try join the new one.')
        assert len(state['participants']) < self._num_max_workers, 'Logic error: joinable rendezvous should always have space left'
        this_rank = len(state['participants'])
        state['participants'].append(this_rank)
        set_ttl: Optional[int] = None
        if len(state['participants']) == self._num_max_workers:
            state['status'] = 'frozen'
            state['keep_alives'] = []
            set_ttl = CONST_ETCD_FROZEN_TTL
        elif len(state['participants']) >= self._num_min_workers:
            set_ttl = CONST_ETCD_JOINABLE_EPHEMERAL_TTL
        try:
            active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=set_ttl)
            return (active_version, this_rank)
        except etcd.EtcdCompareFailed:
            log.info('Join rendezvous CAS unsuccessful, retrying')