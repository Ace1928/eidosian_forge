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
def record_extra_usage_tag(key: TagKey, value: str):
    """Record extra kv usage tag.

    If the key already exists, the value will be overwritten.

    To record an extra tag, first add the key to the TagKey enum and
    then call this function.
    It will make a synchronous call to the internal kv store if the tag is updated.
    """
    key = TagKey.Name(key).lower()
    with _recorded_extra_usage_tags_lock:
        if _recorded_extra_usage_tags.get(key) == value:
            return
        _recorded_extra_usage_tags[key] = value
    if not _internal_kv_initialized():
        return
    _put_extra_usage_tag(key, value)