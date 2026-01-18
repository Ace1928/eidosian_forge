import io
import logging
import queue
import threading
import uuid
import grpc
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private.ray_logging import global_worker_stdstream_dispatcher
from ray._private.worker import print_worker_logs
from ray.util.client.common import CLIENT_SERVER_MAX_THREADS
def log_status_change_thread(log_queue, request_iterator):
    std_handler = StdStreamHandler(log_queue)
    current_handler = None
    root_logger = logging.getLogger('ray')
    default_level = root_logger.getEffectiveLevel()
    try:
        for req in request_iterator:
            if current_handler is not None:
                root_logger.setLevel(default_level)
                root_logger.removeHandler(current_handler)
                std_handler.unregister_global()
            if not req.enabled:
                current_handler = None
                continue
            current_handler = LogstreamHandler(log_queue, req.loglevel)
            std_handler.register_global()
            root_logger.addHandler(current_handler)
            root_logger.setLevel(req.loglevel)
    except grpc.RpcError as e:
        logger.debug(f'closing log thread grpc error reading request_iterator: {e}')
    finally:
        if current_handler is not None:
            root_logger.setLevel(default_level)
            root_logger.removeHandler(current_handler)
            std_handler.unregister_global()
        log_queue.put(None)