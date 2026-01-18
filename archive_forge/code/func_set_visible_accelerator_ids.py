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
def set_visible_accelerator_ids() -> None:
    """Set (CUDA_VISIBLE_DEVICES, ONEAPI_DEVICE_SELECTOR, NEURON_RT_VISIBLE_CORES,
    TPU_VISIBLE_CHIPS , HABANA_VISIBLE_MODULES ,...) environment variables based
    on the accelerator runtime.
    """
    for resource_name, accelerator_ids in ray.get_runtime_context().get_accelerator_ids().items():
        ray._private.accelerators.get_accelerator_manager_for_resource(resource_name).set_current_process_visible_accelerator_ids(accelerator_ids)