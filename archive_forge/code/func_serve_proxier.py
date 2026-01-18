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
def serve_proxier(connection_str: str, address: Optional[str], *, redis_password: Optional[str]=None, session_dir: Optional[str]=None, runtime_env_agent_address: Optional[str]=None):
    if address is not None:
        gcs_cli = GcsClient(address=address)
        ray.experimental.internal_kv._initialize_internal_kv(gcs_cli)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=CLIENT_SERVER_MAX_THREADS), options=GRPC_OPTIONS)
    proxy_manager = ProxyManager(address, session_dir=session_dir, redis_password=redis_password, runtime_env_agent_address=runtime_env_agent_address)
    task_servicer = RayletServicerProxy(None, proxy_manager)
    data_servicer = DataServicerProxy(proxy_manager)
    logs_servicer = LogstreamServicerProxy(proxy_manager)
    ray_client_pb2_grpc.add_RayletDriverServicer_to_server(task_servicer, server)
    ray_client_pb2_grpc.add_RayletDataStreamerServicer_to_server(data_servicer, server)
    ray_client_pb2_grpc.add_RayletLogStreamerServicer_to_server(logs_servicer, server)
    add_port_to_grpc_server(server, connection_str)
    server.start()
    return ClientServerHandle(task_servicer=task_servicer, data_servicer=data_servicer, logs_servicer=logs_servicer, grpc_server=server)