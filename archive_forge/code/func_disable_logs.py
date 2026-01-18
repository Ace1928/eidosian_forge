import sys
import logging
import queue
import threading
import time
import grpc
from typing import TYPE_CHECKING
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.debug import log_once
def disable_logs(self) -> None:
    req = ray_client_pb2.LogSettingsRequest()
    req.enabled = False
    self.request_queue.put(req)
    self.last_req = req