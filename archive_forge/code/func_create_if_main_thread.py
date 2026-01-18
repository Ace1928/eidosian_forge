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
@classmethod
def create_if_main_thread(cls) -> contextlib.AbstractContextManager:
    """Creates a DeferSigint context manager if running on the main thread,
        returns a no-op context manager otherwise.
        """
    if threading.current_thread() == threading.main_thread():
        return cls()
    else:
        return contextlib.nullcontext()