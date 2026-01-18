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
def get_user_temp_dir():
    if 'RAY_TMPDIR' in os.environ:
        return os.environ['RAY_TMPDIR']
    elif sys.platform.startswith('linux') and 'TMPDIR' in os.environ:
        return os.environ['TMPDIR']
    elif sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
        tempdir = os.path.join(os.sep, 'tmp')
    else:
        tempdir = tempfile.gettempdir()
    return tempdir