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
def num_nodes_waiting(self):
    try:
        _, state = self._rdzv_impl.get_rdzv_state()
        if state['status'] == 'final':
            return state['num_workers_waiting']
    except etcd.EtcdKeyNotFound:
        pass
    return 0