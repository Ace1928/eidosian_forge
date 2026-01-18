import json
import logging
import sys
from threading import RLock
from typing import Any, Dict, Optional
import requests
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def get_next_node_id(self):
    with self.lock:
        self._next_node_id += 1
        return self._next_node_id