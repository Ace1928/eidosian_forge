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
def get_log_file_handles(self, name: str, unique: bool=False, create_out: bool=True, create_err: bool=True) -> Tuple[Optional[IO[AnyStr]], Optional[IO[AnyStr]]]:
    """Open log files with partially randomized filenames, returning the
        file handles. If output redirection has been disabled, no files will
        be opened and `(None, None)` will be returned.

        Args:
            name: descriptive string for this log file.
            unique: if true, a counter will be attached to `name` to
                ensure the returned filename is not already used.
            create_out: if True, create a .out file.
            create_err: if True, create a .err file.

        Returns:
            A tuple of two file handles for redirecting optional (stdout, stderr),
            or `(None, None)` if output redirection is disabled.
        """
    if not self.should_redirect_logs():
        return (None, None)
    log_stdout = None
    log_stderr = None
    if create_out:
        log_stdout = open_log(self._get_log_file_name(name, 'out', unique=unique))
    if create_err:
        log_stderr = open_log(self._get_log_file_name(name, 'err', unique=unique))
    return (log_stdout, log_stderr)