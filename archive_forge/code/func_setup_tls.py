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
def setup_tls():
    """Sets up required environment variables for tls"""
    import pytest
    if sys.platform == 'darwin':
        pytest.skip("Cryptography doesn't install in Mac build pipeline")
    cert, key = generate_self_signed_tls_certs()
    temp_dir = tempfile.mkdtemp('ray-test-certs')
    cert_filepath = os.path.join(temp_dir, 'server.crt')
    key_filepath = os.path.join(temp_dir, 'server.key')
    with open(cert_filepath, 'w') as fh:
        fh.write(cert)
    with open(key_filepath, 'w') as fh:
        fh.write(key)
    os.environ['RAY_USE_TLS'] = '1'
    os.environ['RAY_TLS_SERVER_CERT'] = cert_filepath
    os.environ['RAY_TLS_SERVER_KEY'] = key_filepath
    os.environ['RAY_TLS_CA_CERT'] = cert_filepath
    return (key_filepath, cert_filepath, temp_dir)