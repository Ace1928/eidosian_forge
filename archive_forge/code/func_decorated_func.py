import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@functools.wraps(func)
def decorated_func(*args, **kwargs):
    success = False
    try:
        out_stream = io.StringIO()
        sys.stdout = out_stream
        out = func(*args, **kwargs)
        wait_for_condition(lambda: all((string in out_stream.getvalue() for string in strings_to_match)), timeout=timeout_s, retry_interval_ms=1000)
        success = True
        return out
    finally:
        sys.stdout = sys.__stdout__
        if success:
            print('Confirmed expected function stdout. Stdout follows:')
        else:
            print('Did not confirm expected function stdout. Stdout follows:')
        print(out_stream.getvalue())
        out_stream.close()