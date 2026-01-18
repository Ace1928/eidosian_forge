import logging
from typing import Optional
from ray._private import ray_constants
import ray._private.gcs_aio_client
from ray.core.generated.common_pb2 import ErrorType, JobConfig
from ray.core.generated.gcs_pb2 import (
def channel(self):
    return self._channel