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
def set_kill_on_parent_death_linux():
    """Ensures this process dies if its parent dies (fate-sharing).

    Linux-only. Must be called in preexec_fn (i.e. by the child).
    """
    if detect_fate_sharing_support_linux():
        import signal
        PR_SET_PDEATHSIG = 1
        if linux_prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            import ctypes
            raise OSError(ctypes.get_errno(), 'prctl(PR_SET_PDEATHSIG) failed')
    else:
        assert False, 'PR_SET_PDEATHSIG used despite being unavailable'