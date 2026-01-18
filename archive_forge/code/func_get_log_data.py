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
def get_log_data(subscriber, num: int=1000000.0, timeout: float=20, job_id: Optional[str]=None, matcher=None) -> List[dict]:
    deadline = time.time() + timeout
    msgs = []
    while time.time() < deadline and len(msgs) < num:
        logs_data = subscriber.poll(timeout=deadline - time.time())
        if not logs_data:
            break
        if job_id and job_id != logs_data['job']:
            continue
        if matcher and all((not matcher(line) for line in logs_data['lines'])):
            continue
        msgs.append(logs_data)
    return msgs