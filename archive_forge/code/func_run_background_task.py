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
def run_background_task(coroutine: Coroutine) -> asyncio.Task:
    """Schedule a task reliably to the event loop.

    This API is used when you don't want to cache the reference of `asyncio.Task`.
    For example,

    ```
    get_event_loop().create_task(coroutine(*args))
    ```

    The above code doesn't guarantee to schedule the coroutine to the event loops

    When using create_task in a  "fire and forget" way, we should keep the references
    alive for the reliable execution. This API is used to fire and forget
    asynchronous execution.

    https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    """
    task = get_or_create_event_loop().create_task(coroutine)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task