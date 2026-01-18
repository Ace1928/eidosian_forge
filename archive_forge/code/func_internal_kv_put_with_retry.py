import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def internal_kv_put_with_retry(gcs_client, key, value, namespace, num_retries=20):
    if isinstance(key, str):
        key = key.encode()
    if isinstance(value, str):
        value = value.encode()
    if isinstance(namespace, str):
        namespace = namespace.encode()
    error = None
    for _ in range(num_retries):
        try:
            return gcs_client.internal_kv_put(key, value, overwrite=True, namespace=namespace)
        except ray.exceptions.RpcError as e:
            if e.rpc_code in (ray._raylet.GRPC_STATUS_CODE_UNAVAILABLE, ray._raylet.GRPC_STATUS_CODE_UNKNOWN):
                logger.warning(connect_error.format(gcs_client.address))
            else:
                logger.exception('Internal KV Put failed')
            time.sleep(2)
            error = e
    raise error