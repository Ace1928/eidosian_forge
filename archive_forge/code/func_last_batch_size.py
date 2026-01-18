import asyncio
from collections import deque
import logging
import random
from typing import Tuple, List
import grpc
from ray._private.utils import get_or_create_event_loop
import ray._private.gcs_utils as gcs_utils
import ray._private.logging_utils as logging_utils
from ray.core.generated.gcs_pb2 import ErrorTableData
from ray.core.generated import dependency_pb2
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated import gcs_service_pb2
from ray.core.generated import common_pb2
from ray.core.generated import pubsub_pb2
@property
def last_batch_size(self):
    return self._last_batch_size