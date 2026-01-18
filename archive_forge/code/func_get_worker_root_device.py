import logging
import os
import shutil
import tempfile
from typing import Any, Dict
import torch
from packaging.version import Version
import ray
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint
from ray.util import PublicAPI
def get_worker_root_device():
    """Get the first torch device of the current worker if there are multiple."""
    devices = ray.train.torch.get_device()
    if isinstance(devices, list):
        return devices[0]
    else:
        return devices