import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import ray._private.ray_constants as ray_constants
from ray._private.client_mode_hook import (
from ray._private.ray_logging import setup_logger
from ray.job_config import JobConfig
from ray.util.annotations import DeveloperAPI
def is_connected(self, *args, **kwargs):
    return self.get_context().is_connected(*args, **kwargs)