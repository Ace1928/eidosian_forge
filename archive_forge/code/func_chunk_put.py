import math
import logging
import queue
import threading
import warnings
import grpc
from collections import OrderedDict
from typing import Any, Callable, Dict, TYPE_CHECKING, Optional, Union
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.debug import log_once
def chunk_put(req: ray_client_pb2.DataRequest):
    """
    Chunks a put request. Doing this lazily is important for large objects,
    since taking slices of bytes objects does a copy. This means if we
    immediately materialized every chunk of a large object and inserted them
    into the result_queue, we would effectively double the memory needed
    on the client to handle the put.
    """
    request_data = req.put.data
    total_size = len(request_data)
    assert total_size > 0, 'Cannot chunk object with missing data'
    if total_size >= OBJECT_TRANSFER_WARNING_SIZE and log_once('client_object_put_size_warning'):
        size_gb = total_size / 2 ** 30
        warnings.warn(f'Ray Client is attempting to send a {size_gb:.2f} GiB object over the network, which may be slow. Consider serializing the object and using a remote URI to transfer via S3 or Google Cloud Storage instead. Documentation for doing this can be found here: https://docs.ray.io/en/latest/handling-dependencies.html#remote-uris', UserWarning)
    total_chunks = math.ceil(total_size / OBJECT_TRANSFER_CHUNK_SIZE)
    for chunk_id in range(0, total_chunks):
        start = chunk_id * OBJECT_TRANSFER_CHUNK_SIZE
        end = min(total_size, (chunk_id + 1) * OBJECT_TRANSFER_CHUNK_SIZE)
        chunk = ray_client_pb2.PutRequest(client_ref_id=req.put.client_ref_id, data=request_data[start:end], chunk_id=chunk_id, total_chunks=total_chunks, total_size=total_size, owner_id=req.put.owner_id)
        yield ray_client_pb2.DataRequest(req_id=req.req_id, put=chunk)