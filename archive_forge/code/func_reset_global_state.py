import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def reset_global_state():
    global _recorded_library_usages, _recorded_extra_usage_tags
    with _recorded_library_usages_lock:
        _recorded_library_usages = set()
    with _recorded_extra_usage_tags_lock:
        _recorded_extra_usage_tags = dict()