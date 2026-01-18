from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def add_healthz_filter(self, loggers: Optional[Union[str, List[str]]]=None, paths: Optional[Union[str, List[str]]]=None):
    """
        Adds a filter to the logger to filter out healthz requests.

        Parameters
        ----------
        loggers : Optional[Union[str, List[str]]]
            The loggers to add the filter to.
            defaults to ['gunicorn.glogging.Logger', 'uvicorn.access']
        paths : Optional[Union[str, List[str]]]
            The paths to filter out.
            defaults to ['/healthz', '/health']
        """
    if not loggers:
        loggers = ['gunicorn.glogging.Logger', 'uvicorn.access']
    if not paths:
        paths = ['/healthz', '/health']
    if not isinstance(loggers, list):
        loggers = [loggers]
    if not isinstance(paths, list):
        paths = [paths]

    def _healthz_filter(record: logging.LogRecord) -> bool:
        if 'path' in record.args:
            return record.args['path'] not in paths
        return all((path not in record.args for path in paths))
    for logger in loggers:
        logging.getLogger(logger).addFilter(_healthz_filter)