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
def setup_variables(self):
    if not self.is_route_request and (not self.is_health_request):
        service_method_split = self.service_method.split('/')
        self.request = pickle.dumps(self.request)
        self.method_name = service_method_split[-1]
        for key, value in self.context.invocation_metadata():
            if key == 'application':
                self.app_name = value
            elif key == 'request_id':
                self.request_id = value
            elif key == 'multiplexed_model_id':
                self.multiplexed_model_id = value