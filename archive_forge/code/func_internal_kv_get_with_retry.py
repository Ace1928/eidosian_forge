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
def internal_kv_get_with_retry(gcs_client, key, namespace, num_retries=20):
    result = None
    if isinstance(key, str):
        key = key.encode()
    for _ in range(num_retries):
        try:
            result = gcs_client.internal_kv_get(key, namespace)
        except Exception as e:
            if isinstance(e, ray.exceptions.RpcError) and e.rpc_code in (ray._raylet.GRPC_STATUS_CODE_UNAVAILABLE, ray._raylet.GRPC_STATUS_CODE_UNKNOWN):
                logger.warning(connect_error.format(gcs_client.address))
            else:
                logger.exception('Internal KV Get failed')
            result = None
        if result is not None:
            break
        else:
            logger.debug(f'Fetched {key}=None from KV. Retrying.')
            time.sleep(2)
    if not result:
        raise ConnectionError(f"Could not read '{key.decode()}' from GCS. Did GCS start successfully?")
    return result