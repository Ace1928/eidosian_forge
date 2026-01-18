import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def log_to_stderr_filter(record: logging.LogRecord) -> bool:
    """Filters log records based on a parameter in the `extra` dictionary."""
    if not hasattr(record, 'log_to_stderr') or record.log_to_stderr is None:
        return True
    return record.log_to_stderr