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
def set_kill_child_on_death_win32(child_proc):
    """Ensures the child process dies if this process dies (fate-sharing).

    Windows-only. Must be called by the parent, after spawning the child.

    Args:
        child_proc: The subprocess.Popen or subprocess.Handle object.
    """
    if isinstance(child_proc, subprocess.Popen):
        child_proc = child_proc._handle
    assert isinstance(child_proc, subprocess.Handle)
    if detect_fate_sharing_support_win32():
        if not win32_AssignProcessToJobObject(win32_job, int(child_proc)):
            import ctypes
            raise OSError(ctypes.get_last_error(), 'AssignProcessToJobObject() failed')
    else:
        assert False, 'AssignProcessToJobObject used despite being unavailable'