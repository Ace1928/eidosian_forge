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
def get_log_batch(subscriber, num: int, timeout: float=20, job_id: Optional[str]=None, matcher=None) -> List[str]:
    """Gets log batches through GCS subscriber.

    Returns maximum `num` batches of logs. Each batch is a dict that includes
    metadata such as `pid`, `job_id`, and `lines` of log messages.

    If `job_id` or `match` is specified, only returns log batches from `job_id`
    or when `matcher` is true.
    """
    deadline = time.time() + timeout
    batches = []
    while time.time() < deadline and len(batches) < num:
        logs_data = subscriber.poll(timeout=deadline - time.time())
        if not logs_data:
            break
        if job_id and job_id != logs_data['job']:
            continue
        if matcher and (not matcher(logs_data)):
            continue
        batches.append(logs_data)
    return batches