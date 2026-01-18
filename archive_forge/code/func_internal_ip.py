import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def internal_ip(self, node_id):
    return socket.gethostbyname(node_id)