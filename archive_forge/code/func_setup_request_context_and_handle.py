import asyncio
import json
import logging
import os
import pickle
import socket
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type
import grpc
import starlette
import starlette.routing
import uvicorn
from packaging import version
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.types import Receive
import ray
from ray import serve
from ray._private.utils import get_or_create_event_loop
from ray.actor import ActorHandle
from ray.serve._private.common import EndpointInfo, EndpointTag, NodeId, RequestProtocol
from ray.serve._private.constants import (
from ray.serve._private.grpc_util import DummyServicer, create_serve_grpc_server
from ray.serve._private.http_util import (
from ray.serve._private.logging_utils import (
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.proxy_request_response import (
from ray.serve._private.proxy_response_generator import ProxyResponseGenerator
from ray.serve._private.proxy_router import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import call_function_from_import_path
from ray.serve.config import gRPCOptions
from ray.serve.generated.serve_pb2 import HealthzResponse, ListApplicationsResponse
from ray.serve.generated.serve_pb2_grpc import add_RayServeAPIServiceServicer_to_server
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import LoggingConfig
from ray.util import metrics
def setup_request_context_and_handle(self, app_name: str, handle: DeploymentHandle, route_path: str, proxy_request: ProxyRequest) -> Tuple[DeploymentHandle, str]:
    """Setup request context and handle for the request.

        Unpack HTTP request headers and extract info to set up request context and
        handle.
        """
    request_context_info = {'route': route_path, 'app_name': app_name}
    for key, value in proxy_request.headers:
        if key.decode() == SERVE_MULTIPLEXED_MODEL_ID:
            multiplexed_model_id = value.decode()
            handle = handle.options(multiplexed_model_id=multiplexed_model_id)
            request_context_info['multiplexed_model_id'] = multiplexed_model_id
        if key.decode() == 'x-request-id':
            request_context_info['request_id'] = value.decode()
    ray.serve.context._serve_request_context.set(ray.serve.context._RequestContext(**request_context_info))
    return (handle, request_context_info['request_id'])