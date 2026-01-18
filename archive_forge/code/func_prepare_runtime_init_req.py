import atexit
import json
import logging
import socket
import sys
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from itertools import chain
import urllib
from threading import Event, Lock, RLock, Thread
from typing import Callable, Dict, List, Optional, Tuple
import grpc
import psutil
import ray
import ray.core.generated.agent_manager_pb2 as agent_manager_pb2
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
import ray.core.generated.runtime_env_agent_pb2 as runtime_env_agent_pb2
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.parameter import RayParams
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.services import ProcessInfo, start_ray_client_server
from ray._private.tls_utils import add_port_to_grpc_server
from ray._private.utils import detect_fate_sharing_support
from ray.cloudpickle.compat import pickle
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import _get_reconnecting_from_context
def prepare_runtime_init_req(init_request: ray_client_pb2.DataRequest) -> Tuple[ray_client_pb2.DataRequest, JobConfig]:
    """
    Extract JobConfig and possibly mutate InitRequest before it is passed to
    the specific RayClient Server.
    """
    init_type = init_request.WhichOneof('type')
    assert init_type == 'init', f"Received initial message of type {init_type}, not 'init'."
    req = init_request.init
    job_config = JobConfig()
    if req.job_config:
        job_config = pickle.loads(req.job_config)
    new_job_config = ray_client_server_env_prep(job_config)
    modified_init_req = ray_client_pb2.InitRequest(job_config=pickle.dumps(new_job_config), ray_init_kwargs=init_request.init.ray_init_kwargs, reconnect_grace_period=init_request.init.reconnect_grace_period)
    init_request.init.CopyFrom(modified_init_req)
    return (init_request, new_job_config)