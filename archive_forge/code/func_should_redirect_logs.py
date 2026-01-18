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
def should_redirect_logs(self):
    redirect_output = self._ray_params.redirect_output
    if redirect_output is None:
        redirect_output = os.environ.get(ray_constants.LOGGING_REDIRECT_STDERR_ENVIRONMENT_VARIABLE) != '1'
    return redirect_output