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
@classmethod
def get_extra_length(cls, key: str, value: str) -> int:
    """
        Returns the max length of an extra key
        """
    if key not in cls.max_extra_lengths:
        cls.max_extra_lengths[key] = len(key)
    if len(value) > cls.max_extra_lengths[key]:
        cls.max_extra_lengths[key] = len(value)
    return cls.max_extra_lengths[key]