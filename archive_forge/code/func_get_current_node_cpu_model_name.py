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
def get_current_node_cpu_model_name() -> Optional[str]:
    if not sys.platform.startswith('linux'):
        return None
    try:
        '\n        /proc/cpuinfo content example:\n\n        processor\t: 0\n        vendor_id\t: GenuineIntel\n        cpu family\t: 6\n        model\t\t: 85\n        model name\t: Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz\n        stepping\t: 7\n        '
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':')[1].strip()
        return None
    except Exception:
        logger.debug('Failed to get CPU model name', exc_info=True)
        return None