import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def get_webui_url_from_internal_kv():
    assert ray.experimental.internal_kv._internal_kv_initialized()
    webui_url = ray.experimental.internal_kv._internal_kv_get('webui:url', namespace=ray_constants.KV_NAMESPACE_DASHBOARD)
    return ray._private.utils.decode(webui_url) if webui_url is not None else None