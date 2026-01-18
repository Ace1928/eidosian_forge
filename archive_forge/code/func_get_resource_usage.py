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
def get_resource_usage(gcs_address, timeout=10):
    from ray.core.generated import gcs_service_pb2_grpc
    if not gcs_address:
        gcs_address = ray.worker._global_node.gcs_address
    gcs_channel = ray._private.utils.init_grpc_channel(gcs_address, ray_constants.GLOBAL_GRPC_OPTIONS, asynchronous=False)
    gcs_node_resources_stub = gcs_service_pb2_grpc.NodeResourceInfoGcsServiceStub(gcs_channel)
    request = gcs_service_pb2.GetAllResourceUsageRequest()
    response = gcs_node_resources_stub.GetAllResourceUsage(request, timeout=timeout)
    resources_batch_data = response.resource_usage_data
    return resources_batch_data