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
def push_error_to_driver(worker, error_type: str, message: str, job_id: Optional[str]=None):
    """Push an error message to the driver to be printed in the background.

    Args:
        worker: The worker to use.
        error_type: The type of the error.
        message: The message that will be printed in the background
            on the driver.
        job_id: The ID of the driver to push the error message to. If this
            is None, then the message will be pushed to all drivers.
    """
    if job_id is None:
        job_id = ray.JobID.nil()
    assert isinstance(job_id, ray.JobID)
    worker.core_worker.push_error(job_id, error_type, message, time.time())