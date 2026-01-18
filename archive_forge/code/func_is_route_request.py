import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Tuple, Union
import grpc
from starlette.types import Receive, Scope, Send
from ray.actor import ActorHandle
from ray.serve._private.common import StreamingHTTPRequest, gRPCRequest
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import DEFAULT
from ray.serve.grpc_util import RayServegRPCContext
@property
def is_route_request(self) -> bool:
    return self.service_method == '/ray.serve.RayServeAPIService/ListApplications'