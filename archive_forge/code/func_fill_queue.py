from collections import defaultdict
from ray.util.client.server.server_pickler import loads_from_client
import ray
import logging
import grpc
from queue import Queue
import sys
from typing import Any, Dict, Iterator, TYPE_CHECKING, Union
from threading import Event, Lock, Thread
import time
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.client import CURRENT_PROTOCOL_VERSION
from ray.util.debug import log_once
from ray._private.client_mode_hook import disable_client_hook
def fill_queue(grpc_input_generator: Iterator[ray_client_pb2.DataRequest], output_queue: 'Queue[Union[ray_client_pb2.DataRequest, ray_client_pb2.DataResponse]]') -> None:
    """
    Pushes incoming requests to a shared output_queue.
    """
    try:
        for req in grpc_input_generator:
            output_queue.put(req)
    except grpc.RpcError as e:
        logger.debug(f'closing dataservicer reader thread grpc error reading request_iterator: {e}')
    finally:
        output_queue.put(None)