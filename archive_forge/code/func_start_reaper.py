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
def start_reaper(fate_share=None):
    """Start the reaper process.

    This is a lightweight process that simply
    waits for its parent process to die and then terminates its own
    process group. This allows us to ensure that ray processes are always
    terminated properly so long as that process itself isn't SIGKILLed.

    Returns:
        ProcessInfo for the process that was started.
    """
    try:
        if sys.platform != 'win32':
            os.setpgrp()
    except OSError as e:
        errcode = e.errno
        if errcode == errno.EPERM and os.getpgrp() == os.getpid():
            pass
        else:
            logger.warning(f'setpgrp failed, processes may not be cleaned up properly: {e}.')
            return None
    reaper_filepath = os.path.join(RAY_PATH, RAY_PRIVATE_DIR, 'ray_process_reaper.py')
    command = [sys.executable, '-u', reaper_filepath]
    process_info = start_ray_process(command, ray_constants.PROCESS_TYPE_REAPER, pipe_stdin=True, fate_share=fate_share)
    return process_info