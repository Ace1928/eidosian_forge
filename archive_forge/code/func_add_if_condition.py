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
def add_if_condition(self, name: str, condition: Union[Callable, bool], level: Optional[Union[str, int]]='INFO'):
    """
        Adds a condition to the logger
        """
    self.conditions[name] = (condition, self._get_level(level))