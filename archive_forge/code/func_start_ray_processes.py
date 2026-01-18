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
def start_ray_processes(self):
    """Start all of the processes on the node."""
    logger.debug(f'Process STDOUT and STDERR is being redirected to {self._logs_dir}.')
    if not self.head:
        gcs_options = ray._raylet.GcsClientOptions.from_gcs_address(self.gcs_address)
        global_state = ray._private.state.GlobalState()
        global_state._initialize_global_state(gcs_options)
        new_config = global_state.get_system_config()
        assert self._config.items() <= new_config.items(), f'The system config from GCS is not a superset of the local system config. There might be a configuration inconsistency issue between the head node and non-head nodes. Local system config: {self._config}, GCS system config: {new_config}'
        self._config = new_config
    self.destroy_external_storage()
    resource_spec = self.get_resource_spec()
    plasma_directory, object_store_memory = ray._private.services.determine_plasma_store_config(resource_spec.object_store_memory, plasma_directory=self._ray_params.plasma_directory, huge_pages=self._ray_params.huge_pages)
    self.start_raylet(plasma_directory, object_store_memory)
    if self._ray_params.include_log_monitor:
        self.start_log_monitor()