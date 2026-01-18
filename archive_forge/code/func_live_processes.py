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
def live_processes(self):
    """Return a list of the live processes.

        Returns:
            A list of the live processes.
        """
    result = []
    for process_type, process_infos in self.all_processes.items():
        for process_info in process_infos:
            if process_info.process.poll() is None:
                result.append((process_type, process_info.process))
    return result