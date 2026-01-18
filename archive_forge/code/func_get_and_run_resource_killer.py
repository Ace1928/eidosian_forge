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
def get_and_run_resource_killer(resource_killer_cls, kill_interval_s, namespace=None, lifetime=None, no_start=False, max_to_kill=2, kill_delay_s=0, kill_filter_fn=None):
    assert ray.is_initialized(), 'The API is only available when Ray is initialized.'
    name = resource_killer_cls.__ray_actor_class__.__name__
    head_node_id = ray.get_runtime_context().get_node_id()
    resource_killer = resource_killer_cls.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False), namespace=namespace, name=name, lifetime=lifetime).remote(head_node_id, kill_interval_s=kill_interval_s, max_to_kill=max_to_kill, kill_filter_fn=kill_filter_fn)
    print(f'Waiting for {name} to be ready...')
    ray.get(resource_killer.ready.remote())
    print(f'{name} is ready now.')
    if not no_start:
        time.sleep(kill_delay_s)
        resource_killer.run.remote()
    return resource_killer