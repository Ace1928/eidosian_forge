import colorama
from dataclasses import dataclass
import logging
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Set, Tuple, Any, Optional
import ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray._private.ray_constants import (
from ray.util.debug import log_once
def setup_component_logger(*, logging_level, logging_format, log_dir, filename, max_bytes, backup_count, logger_name=None, propagate=True):
    """Configure the logger that is used for Ray's python components.

    For example, it should be used for monitor, dashboard, and log monitor.
    The only exception is workers. They use the different logging config.

    Ray's python components generally should not write to stdout/stderr, because
    messages written there will be redirected to the head node. For deployments where
    there may be thousands of workers, this would create unacceptable levels of log
    spam. For this reason, we disable the "ray" logger's handlers, and enable
    propagation so that log messages that actually do need to be sent to the head node
    can reach it.

    Args:
        logging_level: Logging level in string or logging enum.
        logging_format: Logging format string.
        log_dir: Log directory path. If empty, logs will go to
            stderr.
        filename: Name of the file to write logs. If empty, logs will go
            to stderr.
        max_bytes: Same argument as RotatingFileHandler's maxBytes.
        backup_count: Same argument as RotatingFileHandler's backupCount.
        logger_name: Used to create or get the correspoding
            logger in getLogger call. It will get the root logger by default.
        propagate: Whether to propagate the log to the parent logger.
    Returns:
        the created or modified logger.
    """
    ray._private.log.clear_logger('ray')
    logger = logging.getLogger(logger_name)
    if type(logging_level) is str:
        logging_level = logging.getLevelName(logging_level.upper())
    if not filename or not log_dir:
        handler = logging.StreamHandler()
    else:
        handler = logging.handlers.RotatingFileHandler(os.path.join(log_dir, filename), maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(logging_level)
    logger.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(logging_format))
    logger.addHandler(handler)
    logger.propagate = propagate
    return logger