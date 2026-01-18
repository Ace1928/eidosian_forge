import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional
import yaml
import ray
from ray._private.dict import deep_update
from ray.autoscaler._private.fake_multi_node.node_provider import (
from ray.util.queue import Empty, Queue
def remote_execution_api(self) -> 'RemoteAPI':
    """Create an object to control cluster state from within the cluster."""
    self._execution_queue = Queue(actor_options={'num_cpus': 0})
    stop_event = self._execution_event

    def entrypoint():
        while not stop_event.is_set():
            try:
                cmd, kwargs = self._execution_queue.get(timeout=1)
            except Empty:
                continue
            if cmd == 'kill_node':
                self.kill_node(**kwargs)
    self._execution_thread = threading.Thread(target=entrypoint)
    self._execution_thread.start()
    return RemoteAPI(self._execution_queue)