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
def send_request_id(self, request_id: str):
    self.ray_serve_grpc_context.set_trailing_metadata([('request_id', request_id)])