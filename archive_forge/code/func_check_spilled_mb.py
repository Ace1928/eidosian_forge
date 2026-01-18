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
def check_spilled_mb(address, spilled=None, restored=None, fallback=None):

    def ok():
        s = memory_summary(address=address['address'], stats_only=True)
        print(s)
        if restored:
            if 'Restored {} MiB'.format(restored) not in s:
                return False
        elif 'Restored' in s:
            return False
        if spilled:
            if not isinstance(spilled, list):
                spilled_lst = [spilled]
            else:
                spilled_lst = spilled
            found = False
            for n in spilled_lst:
                if 'Spilled {} MiB'.format(n) in s:
                    found = True
            if not found:
                return False
        elif 'Spilled' in s:
            return False
        if fallback:
            if 'Plasma filesystem mmap usage: {} MiB'.format(fallback) not in s:
                return False
        elif 'Plasma filesystem mmap usage:' in s:
            return False
        return True
    wait_for_condition(ok, timeout=3, retry_interval_ms=1000)