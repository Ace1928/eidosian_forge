import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, IO, AnyStr
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
from ray._private import storage
from ray._raylet import GcsClient, get_session_key_from_storage
from ray._private.resource_spec import ResourceSpec
from ray._private.services import serialize_config, get_address
from ray._private.utils import open_log, try_to_create_directory, try_to_symlink
def start_head_processes(self):
    """Start head processes on the node."""
    logger.debug(f'Process STDOUT and STDERR is being redirected to {self._logs_dir}.')
    assert self._gcs_address is None
    assert self._gcs_client is None
    self.start_gcs_server()
    assert self.get_gcs_client() is not None
    self._write_cluster_info_to_kv()
    if not self._ray_params.no_monitor:
        self.start_monitor()
    if self._ray_params.ray_client_server_port:
        self.start_ray_client_server()
    if self._ray_params.include_dashboard is None:
        raise_on_api_server_failure = False
    else:
        raise_on_api_server_failure = self._ray_params.include_dashboard
    self.start_api_server(include_dashboard=self._ray_params.include_dashboard, raise_on_failure=raise_on_api_server_failure)