import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._private.tls_utils
import ray.cloudpickle as cloudpickle
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private.ray_constants import DEFAULT_CLIENT_RECONNECT_GRACE_PERIOD
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.cloudpickle.compat import pickle
from ray.exceptions import GetTimeoutError
from ray.job_config import JobConfig
from ray.util.client.client_pickler import dumps_from_client, loads_from_server
from ray.util.client.common import (
from ray.util.client.dataclient import DataClient
from ray.util.client.logsclient import LogstreamClient
from ray.util.debug import log_once
def internal_kv_list(self, prefix: bytes, namespace: Optional[bytes]) -> List[bytes]:
    try:
        req = ray_client_pb2.KVListRequest(prefix=prefix, namespace=namespace)
        return self._call_stub('KVList', req, metadata=self.metadata).keys
    except grpc.RpcError as e:
        raise decode_exception(e)