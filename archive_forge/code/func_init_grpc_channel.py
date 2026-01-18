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
def init_grpc_channel(address: str, options: Optional[Sequence[Tuple[str, Any]]]=None, asynchronous: bool=False):
    import grpc
    try:
        from grpc import aio as aiogrpc
    except ImportError:
        from grpc.experimental import aio as aiogrpc
    from ray._private.tls_utils import load_certs_from_env
    grpc_module = aiogrpc if asynchronous else grpc
    options = options or []
    options_dict = dict(options)
    options_dict['grpc.keepalive_time_ms'] = options_dict.get('grpc.keepalive_time_ms', ray._config.grpc_client_keepalive_time_ms())
    options_dict['grpc.keepalive_timeout_ms'] = options_dict.get('grpc.keepalive_timeout_ms', ray._config.grpc_client_keepalive_timeout_ms())
    options = options_dict.items()
    if os.environ.get('RAY_USE_TLS', '0').lower() in ('1', 'true'):
        server_cert_chain, private_key, ca_cert = load_certs_from_env()
        credentials = grpc.ssl_channel_credentials(certificate_chain=server_cert_chain, private_key=private_key, root_certificates=ca_cert)
        channel = grpc_module.secure_channel(address, credentials, options=options)
    else:
        channel = grpc_module.insecure_channel(address, options=options)
    return channel