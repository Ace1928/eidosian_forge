import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def set_node_tags(self, node_id, tags):
    with self.state.file_lock:
        info = self.state.get()[node_id]
        info['tags'].update(tags)
        self.state.put(node_id, info)