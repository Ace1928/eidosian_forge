import datetime
import sys
from typing import (
import warnings
import duet
import proto
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.protobuf import any_pb2, field_mask_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from cirq._compat import cached_property
from cirq._compat import deprecated_parameter
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine import stream_manager
@cached_property
def grpc_client(self) -> quantum.QuantumEngineServiceAsyncClient:
    """Creates an async grpc client for the quantum engine service."""

    async def make_client():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return quantum.QuantumEngineServiceAsyncClient(**self._service_args)
    return self._executor.submit(make_client).result()