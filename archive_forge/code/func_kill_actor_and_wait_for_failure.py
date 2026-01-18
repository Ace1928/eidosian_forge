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
def kill_actor_and_wait_for_failure(actor, timeout=10, retry_interval_ms=100):
    actor_id = actor._actor_id.hex()
    current_num_restarts = ray._private.state.actors(actor_id)['NumRestarts']
    ray.kill(actor)
    start = time.time()
    while time.time() - start <= timeout:
        actor_status = ray._private.state.actors(actor_id)
        if actor_status['State'] == convert_actor_state(gcs_utils.ActorTableData.DEAD) or actor_status['NumRestarts'] > current_num_restarts:
            return
        time.sleep(retry_interval_ms / 1000.0)
    raise RuntimeError('It took too much time to kill an actor: {}'.format(actor_id))