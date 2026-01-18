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
def service_handler_factory(self, service_method: str, stream: bool) -> Callable:

    def set_grpc_code_and_details(context: grpc._cython.cygrpc._ServicerContext, status: ResponseStatus):
        if not context.code():
            context.set_code(status.code)
        if not context.details():
            context.set_details(status.message)

    async def unary_unary(request_proto: Any, context: grpc._cython.cygrpc._ServicerContext) -> bytes:
        """Entry point of the gRPC proxy unary request.

            This method is called by the gRPC server when a unary request is received.
            It wraps the request in a ProxyRequest object and calls proxy_request.
            The return value is serialized user defined protobuf bytes.
            """
        proxy_request = gRPCProxyRequest(request_proto=request_proto, context=context, service_method=service_method, stream=False)
        status = None
        response = None
        async for message in self.proxy_request(proxy_request=proxy_request):
            if isinstance(message, ResponseStatus):
                status = message
            else:
                response = message
        set_grpc_code_and_details(context, status)
        return response

    async def unary_stream(request_proto: Any, context: grpc._cython.cygrpc._ServicerContext) -> Generator[bytes, None, None]:
        """Entry point of the gRPC proxy streaming request.

            This method is called by the gRPC server when a streaming request is
            received. It wraps the request in a ProxyRequest object and calls
            proxy_request. The return value is a generator of serialized user defined
            protobuf bytes.
            """
        proxy_request = gRPCProxyRequest(request_proto=request_proto, context=context, service_method=service_method, stream=True)
        status = None
        async for message in self.proxy_request(proxy_request=proxy_request):
            if isinstance(message, ResponseStatus):
                status = message
            else:
                yield message
        set_grpc_code_and_details(context, status)
    return unary_stream if stream else unary_unary