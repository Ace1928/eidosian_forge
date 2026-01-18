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
def find_available_port(start, end, port_num=1):
    ports = []
    for _ in range(port_num):
        random_port = 0
        with socket.socket() as s:
            s.bind(('', 0))
            random_port = s.getsockname()[1]
        if random_port >= start and random_port <= end and (random_port not in ports):
            ports.append(random_port)
            continue
        for port in range(start, end + 1):
            if port in ports:
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                ports.append(port)
                break
            except OSError:
                pass
    if len(ports) != port_num:
        raise RuntimeError(f"Can't find {port_num} available port from {start} to {end}.")
    return ports