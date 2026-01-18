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
def check_version_info(self):
    """Check if the Python and Ray version of this process matches that in GCS.

        This will be used to detect if workers or drivers are started using
        different versions of Python, or Ray.

        Raises:
            Exception: An exception is raised if there is a version mismatch.
        """
    import ray._private.usage.usage_lib as ray_usage_lib
    cluster_metadata = ray_usage_lib.get_cluster_metadata(self.get_gcs_client())
    if cluster_metadata is None:
        cluster_metadata = ray_usage_lib.get_cluster_metadata(self.get_gcs_client())
    if not cluster_metadata:
        return
    ray._private.utils.check_version_info(cluster_metadata)