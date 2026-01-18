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
def get_call_location(back: int=1):
    """
    Get the location (filename and line number) of a function caller, `back`
    frames up the stack.

    Args:
        back: The number of frames to go up the stack, not including this
            function.
    """
    stack = inspect.stack()
    try:
        frame = stack[back + 1]
        return f'{frame.filename}:{frame.lineno}'
    except IndexError:
        return 'UNKNOWN'