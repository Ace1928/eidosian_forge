import copy
import json
import logging
import os
import subprocess
import sys
import time
from threading import RLock
from types import ModuleType
from typing import Any, Dict, Optional
import yaml
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.fake_multi_node.command_runner import (
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def host_dir(container_dir: str):
    """Replace local dir with potentially different host dir.

    E.g. in docker-in-docker environments, the host dir might be
    different to the mounted directory in the container.

    This method will do a simple global replace to adjust the paths.
    """
    ray_tempdir = os.environ.get('RAY_TEMPDIR', None)
    ray_hostdir = os.environ.get('RAY_HOSTDIR', None)
    if not ray_tempdir or not ray_hostdir:
        return container_dir
    return container_dir.replace(ray_tempdir, ray_hostdir)