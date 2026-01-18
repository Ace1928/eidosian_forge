import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def create_node(self, node_config, tags, count):
    """Creates min(count, currently available) nodes."""
    node_type = tags[TAG_RAY_NODE_KIND]
    with self.state.file_lock:
        workers = self.state.get()
        for node_id, info in workers.items():
            if info['state'] == 'terminated' and (self.use_coordinator or info['tags'][TAG_RAY_NODE_KIND] == node_type):
                info['tags'] = tags
                info['state'] = 'running'
                self.state.put(node_id, info)
                count = count - 1
                if count == 0:
                    return