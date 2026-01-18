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
def create_global_logger(name: Optional[str]='lazyops', level: Union[str, int]='INFO', format: Optional[Callable]=None, filter: Optional[Callable]=None, handlers: Optional[List[logging.Handler]]=None, settings: Optional['BaseSettings']=None, **kwargs) -> Logger:
    """
    Creates the global logger
    """
    global _logger_contexts
    try:
        _logger = Logger(core=_Core(), exception=None, depth=0, record=False, lazy=False, colors=True, raw=False, capture=True, patcher=None, extra={})
    except Exception as e:
        _logger = Logger(core=_Core(), exception=None, depth=0, record=False, lazy=False, colors=False, raw=False, capture=True, patchers=[], extra={})
    _logger.name = 'lazyops'
    _logger.is_global = True
    dev_level = _logger.level(name='DEV', no=19, color='<blue>', icon='@')
    if _defaults.LOGURU_AUTOINIT and sys.stderr:
        _logger.add(sys.stderr)
    _atexit.register(_logger.remove)
    _logger.remove()
    _logger.add_if_condition('dev', _logger._is_dev_condition)
    logging.basicConfig(handlers=handlers or [InterceptHandler()], level=0)
    _logger.add(sys.stdout, enqueue=True, backtrace=True, colorize=True, level=level, format=format if format is not None else LoggerFormatter.default_formatter, filter=filter if filter is not None else _logger._filter, **kwargs)
    if settings:
        _logger.settings = settings
    _logger_contexts[name] = _logger
    return _logger