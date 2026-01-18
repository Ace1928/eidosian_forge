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
@property
def colored_opts(self):
    """
        Returns the colored options
        """
    if not self._colored_opts:
        exception, depth, record, lazy, colors, raw, capture, patchers, extra = self._options
        self._colored_opts = (exception, depth, record, lazy, True, raw, capture, patchers, extra)
    return self._colored_opts