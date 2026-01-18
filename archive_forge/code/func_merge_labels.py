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
def merge_labels(env_override_labels, params_labels):
    """Merges two dictionaries, picking from the
            first in the event of a conflict. Also emit a warning on every
            conflict.
            """
    result = params_labels.copy()
    result.update(env_override_labels)
    for key in set(env_override_labels.keys()).intersection(set(params_labels.keys())):
        if params_labels[key] != env_override_labels[key]:
            logger.warning(f'Autoscaler is overriding your label:{key}: {params_labels[key]} to {key}: {env_override_labels[key]}.')
    return result