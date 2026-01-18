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
def kill_all_processes(self, check_alive=True, allow_graceful=False, wait=False):
    """Kill all of the processes.

        Note that This is slower than necessary because it calls kill, wait,
        kill, wait, ... instead of kill, kill, ..., wait, wait, ...

        Args:
            check_alive: Raise an exception if any of the processes were
                already dead.
            wait: If true, then this method will not return until the
                process in question has exited.
        """
    if ray_constants.PROCESS_TYPE_RAYLET in self.all_processes:
        self._kill_process_type(ray_constants.PROCESS_TYPE_RAYLET, check_alive=check_alive, allow_graceful=allow_graceful, wait=wait)
    if ray_constants.PROCESS_TYPE_GCS_SERVER in self.all_processes:
        self._kill_process_type(ray_constants.PROCESS_TYPE_GCS_SERVER, check_alive=check_alive, allow_graceful=allow_graceful, wait=wait)
    for process_type in list(self.all_processes.keys()):
        if process_type != ray_constants.PROCESS_TYPE_REAPER:
            self._kill_process_type(process_type, check_alive=check_alive, allow_graceful=allow_graceful, wait=wait)
    if ray_constants.PROCESS_TYPE_REAPER in self.all_processes:
        self._kill_process_type(ray_constants.PROCESS_TYPE_REAPER, check_alive=check_alive, allow_graceful=allow_graceful, wait=wait)