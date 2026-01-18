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
def start_log_monitor(self):
    """Start the log monitor."""
    _, stderr_file = self.get_log_file_handles('log_monitor', unique=True, create_out=False)
    process_info = ray._private.services.start_log_monitor(self.get_session_dir_path(), self._logs_dir, self.gcs_address, fate_share=self.kernel_fate_share, max_bytes=self.max_bytes, backup_count=self.backup_count, redirect_logging=self.should_redirect_logs(), stdout_file=stderr_file, stderr_file=stderr_file)
    assert ray_constants.PROCESS_TYPE_LOG_MONITOR not in self.all_processes
    self.all_processes[ray_constants.PROCESS_TYPE_LOG_MONITOR] = [process_info]