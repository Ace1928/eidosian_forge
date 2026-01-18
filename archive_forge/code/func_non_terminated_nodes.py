import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def non_terminated_nodes(self, tag_filters):
    workers = self.state.get()
    matching_ips = []
    for worker_ip, info in workers.items():
        if info['state'] == 'terminated':
            continue
        ok = True
        for k, v in tag_filters.items():
            if info['tags'].get(k) != v:
                ok = False
                break
        if ok:
            matching_ips.append(worker_ip)
    return matching_ips