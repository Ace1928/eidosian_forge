import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
@contextlib.contextmanager
def filter_logs(logger: logging.Logger, filters: Sequence[logging.Filter]) -> Generator[None, None, None]:
    """Temporarily adds specified filters to a logger.

    Parameters:
    - logger: The logger to which the filters will be added.
    - filters: A sequence of logging.Filter objects to be temporarily added
        to the logger.
    """
    with _FILTER_LOCK:
        for filter in filters:
            logger.addFilter(filter)
    try:
        yield
    finally:
        with _FILTER_LOCK:
            for filter in filters:
                try:
                    logger.removeFilter(filter)
                except BaseException:
                    _LOGGER.warning('Failed to remove filter')