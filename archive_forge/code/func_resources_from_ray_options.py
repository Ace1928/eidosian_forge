import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def resources_from_ray_options(options_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Determine a task's resource requirements.

    Args:
        options_dict: The dictionary that contains resources requirements.

    Returns:
        A dictionary of the resource requirements for the task.
    """
    resources = (options_dict.get('resources') or {}).copy()
    if 'CPU' in resources or 'GPU' in resources:
        raise ValueError("The resources dictionary must not contain the key 'CPU' or 'GPU'")
    elif 'memory' in resources or 'object_store_memory' in resources:
        raise ValueError("The resources dictionary must not contain the key 'memory' or 'object_store_memory'")
    num_cpus = options_dict.get('num_cpus')
    num_gpus = options_dict.get('num_gpus')
    memory = options_dict.get('memory')
    object_store_memory = options_dict.get('object_store_memory')
    accelerator_type = options_dict.get('accelerator_type')
    if num_cpus is not None:
        resources['CPU'] = num_cpus
    if num_gpus is not None:
        resources['GPU'] = num_gpus
    if memory is not None:
        resources['memory'] = int(memory)
    if object_store_memory is not None:
        resources['object_store_memory'] = object_store_memory
    if accelerator_type is not None:
        resources[f'{ray_constants.RESOURCE_CONSTRAINT_PREFIX}{accelerator_type}'] = 0.001
    return resources