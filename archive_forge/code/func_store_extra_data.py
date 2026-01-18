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
def store_extra_data(self, rdzv_version, key, value):
    node = self.get_path(f'/rdzv/v_{rdzv_version}/extra_data')
    try:
        extra_data = self.client.write(key=node, value=json.dumps({key: value}), prevExist=False)
        return
    except etcd.EtcdAlreadyExist:
        pass
    while True:
        extra_data = self.client.get(node)
        new_extra_data_value = json.loads(extra_data.value)
        new_extra_data_value[key] = value
        try:
            extra_data = self.client.test_and_set(key=node, value=json.dumps(new_extra_data_value), prev_value=extra_data.value)
            return
        except etcd.EtcdCompareFailed:
            log.info('Store extra_data CAS unsuccessful, retrying')
            time.sleep(0.1)