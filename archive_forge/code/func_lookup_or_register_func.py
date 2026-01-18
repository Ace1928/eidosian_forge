import base64
import functools
import gc
import inspect
import json
import logging
import math
import pickle
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Set, Union
import grpc
import ray
import ray._private.state
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray import cloudpickle
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.ray_constants import env_integer
from ray._private.ray_logging import setup_logger
from ray._private.services import canonicalize_bootstrap_address_or_die
from ray._private.tls_utils import add_port_to_grpc_server
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import DataServicer
from ray.util.client.server.logservicer import LogstreamServicer
from ray.util.client.server.proxier import serve_proxier
from ray.util.client.server.server_pickler import dumps_from_server, loads_from_client
from ray.util.client.server.server_stubs import current_server
def lookup_or_register_func(self, id: bytes, client_id: str, options: Optional[Dict]) -> ray.remote_function.RemoteFunction:
    with disable_client_hook():
        if id not in self.function_refs:
            funcref = self.object_refs[client_id][id]
            func = ray.get(funcref)
            if not inspect.isfunction(func):
                raise Exception("Attempting to register function that isn't a function.")
            if options is None or len(options) == 0:
                self.function_refs[id] = ray.remote(func)
            else:
                self.function_refs[id] = ray.remote(**options)(func)
    return self.function_refs[id]