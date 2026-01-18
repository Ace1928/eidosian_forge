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
def mute_module(self, module: str, **kwargs):
    """
        Efectively mutes a module
            
            - `module`: The module to mute
        """

    def _mute_filter(record: logging.LogRecord) -> bool:
        return record.module == module
    logging.getLogger(module).addFilter(_mute_filter)