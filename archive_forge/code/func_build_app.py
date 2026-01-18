import asyncio
import json
import logging
import time
from random import random
from typing import Callable, Dict
import aiohttp
import numpy as np
import pandas as pd
from grpc import aio
from starlette.requests import Request
import ray
from ray import serve
from ray.serve._private.common import RequestProtocol
from ray.serve.config import gRPCOptions
from ray.serve.generated import serve_pb2, serve_pb2_grpc
from ray.serve.handle import RayServeHandle
def build_app(num_replicas: int, max_concurrent_queries: int, data_size: int):

    @serve.deployment(max_concurrent_queries=1000)
    class DataPreprocessing:

        def __init__(self, handle: RayServeHandle):
            self._handle = handle
            logging.getLogger('ray.serve').setLevel(logging.WARNING)

        def normalize(self, raw: np.ndarray) -> np.ndarray:
            return (raw - np.min(raw)) / (np.max(raw) - np.min(raw) + DELTA)

        async def __call__(self, req: Request):
            """HTTP entrypoint.

            It parses the request, normalize the data, and send to model for inference.
            """
            body = json.loads(await req.body())
            raw = np.asarray(body['nums'])
            processed = self.normalize(raw)
            model_output_ref = await self._handle.remote(processed)
            return await model_output_ref

        async def grpc_call(self, raq_data):
            """gRPC entrypoint.

            It parses the request, normalize the data, and send to model for inference.
            """
            raw = np.asarray(raq_data.nums)
            processed = self.normalize(raw)
            model_output_ref = await self._handle.remote(processed)
            return serve_pb2.ModelOutput(output=await model_output_ref)

    @serve.deployment(num_replicas=num_replicas, max_concurrent_queries=max_concurrent_queries)
    class ModelInference:

        def __init__(self):
            logging.getLogger('ray.serve').setLevel(logging.WARNING)
            self._model = np.random.randn(data_size, data_size)

        async def __call__(self, processed: np.ndarray) -> float:
            model_output = np.dot(processed, self._model)
            return sum(model_output)
    return DataPreprocessing.bind(ModelInference.bind())