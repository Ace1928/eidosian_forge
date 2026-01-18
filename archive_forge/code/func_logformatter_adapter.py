from __future__ import annotations
import logging
import sys
import warnings
from logging.config import dictConfig
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union, cast
from twisted.python import log as twisted_log
from twisted.python.failure import Failure
import scrapy
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.settings import Settings
from scrapy.utils.versions import scrapy_components_versions
def logformatter_adapter(logkws: dict) -> Tuple[int, str, dict]:
    """
    Helper that takes the dictionary output from the methods in LogFormatter
    and adapts it into a tuple of positional arguments for logger.log calls,
    handling backward compatibility as well.
    """
    if not {'level', 'msg', 'args'} <= set(logkws):
        warnings.warn('Missing keys in LogFormatter method', ScrapyDeprecationWarning)
    if 'format' in logkws:
        warnings.warn('`format` key in LogFormatter methods has been deprecated, use `msg` instead', ScrapyDeprecationWarning)
    level = logkws.get('level', logging.INFO)
    message = logkws.get('format', logkws.get('msg'))
    args = logkws if not logkws.get('args') else logkws['args']
    return (level, message, args)