import copy
import logging
import time
from functools import wraps
from threading import RLock
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import googleapiclient
from ray.autoscaler._private.gcp.config import (
from ray.autoscaler._private.gcp.node import GCPTPU  # noqa
from ray.autoscaler._private.gcp.node import (
from ray.autoscaler._private.gcp.tpu_command_runner import TPUCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
@wraps(method)
def method_with_retries(self, *args, **kwargs):
    try_count = 0
    while try_count < max_tries:
        try:
            return method(self, *args, **kwargs)
        except BrokenPipeError:
            logger.warning('Caught a BrokenPipeError. Retrying.')
            try_count += 1
            if try_count < max_tries:
                self._construct_clients()
                time.sleep(backoff_s)
            else:
                raise