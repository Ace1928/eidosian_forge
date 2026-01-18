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
def fetch_prometheus(prom_addresses):
    components_dict = {}
    metric_names = set()
    metric_samples = []
    for address in prom_addresses:
        if address not in components_dict:
            components_dict[address] = set()
    for address, response in fetch_raw_prometheus(prom_addresses):
        for line in response.split('\n'):
            for family in text_string_to_metric_families(line):
                for sample in family.samples:
                    metric_names.add(sample.name)
                    metric_samples.append(sample)
                    if 'Component' in sample.labels:
                        components_dict[address].add(sample.labels['Component'])
    return (components_dict, metric_names, metric_samples)