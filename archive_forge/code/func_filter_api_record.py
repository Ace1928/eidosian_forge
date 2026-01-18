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
def filter_api_record(record: logging.LogRecord) -> bool:
    """
        Filter out health checks and other unwanted logs
        """
    if routes:
        for route in routes:
            if route in record.args:
                return False
    if status_codes:
        for sc in status_codes:
            if sc in record.args:
                return False
    return True