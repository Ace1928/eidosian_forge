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
@property
def gcs_port(self):
    return self._cluster_config.get('provider', {}).get('host_gcs_port', FAKE_DOCKER_DEFAULT_GCS_PORT)