import logging
from typing import Optional
from ray._private import ray_constants
import ray._private.gcs_aio_client
from ray.core.generated.common_pb2 import ErrorType, JobConfig
from ray.core.generated.gcs_pb2 import (
def create_gcs_channel(address: str, aio=False):
    """Returns a GRPC channel to GCS.

    Args:
        address: GCS address string, e.g. ip:port
        aio: Whether using grpc.aio
    Returns:
        grpc.Channel or grpc.aio.Channel to GCS
    """
    from ray._private.utils import init_grpc_channel
    return init_grpc_channel(address, options=_GRPC_OPTIONS, asynchronous=aio)