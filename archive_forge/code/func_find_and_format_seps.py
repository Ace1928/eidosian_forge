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
def find_and_format_seps(msg: str) -> str:
    """
    Find any |a,b,c| and format them |a||b||c|

    ex:
      |em,b,u| -> |em||b||u|
      |em,b| -> |em||b|
      
    """
    for sep_match in re.finditer('\\|\\w+,(\\w+,*)+\\|', msg):
        s = sep_match.group()
        if len(s) >= 10:
            continue
        msg = msg.replace(s, '||'.join(s.split(',')))
    return msg