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
def compute_version_info():
    """Compute the versions of Python, and Ray.

    Returns:
        A tuple containing the version information.
    """
    ray_version = ray.__version__
    python_version = '.'.join(map(str, sys.version_info[:3]))
    return (ray_version, python_version)