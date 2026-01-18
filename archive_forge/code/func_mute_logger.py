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
def mute_logger(self, modules: Optional[Union[str, List[str]]], level: str='WARNING'):
    """
        Helper to mute a logger from another module.
        """
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        logging.getLogger(module).setLevel(logging.getLevelName(level))