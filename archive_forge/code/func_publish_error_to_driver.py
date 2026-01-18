import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def publish_error_to_driver(error_type: str, message: str, gcs_publisher, job_id=None, num_retries=None):
    """Push an error message to the driver to be printed in the background.

    Normally the push_error_to_driver function should be used. However, in some
    instances, the raylet client is not available, e.g., because the
    error happens in Python before the driver or worker has connected to the
    backend processes.

    Args:
        error_type: The type of the error.
        message: The message that will be printed in the background
            on the driver.
        gcs_publisher: The GCS publisher to use.
        job_id: The ID of the driver to push the error message to. If this
            is None, then the message will be pushed to all drivers.
    """
    if job_id is None:
        job_id = ray.JobID.nil()
    assert isinstance(job_id, ray.JobID)
    try:
        gcs_publisher.publish_error(job_id.hex().encode(), error_type, message, job_id, num_retries)
    except Exception:
        logger.exception(f'Failed to publish error: {message} [type {error_type}]')