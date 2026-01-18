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
def failure_to_exc_info(failure: Failure) -> Optional[Tuple[Type[BaseException], BaseException, Optional[TracebackType]]]:
    """Extract exc_info from Failure instances"""
    if isinstance(failure, Failure):
        assert failure.type
        assert failure.value
        return (failure.type, failure.value, cast(Optional[TracebackType], failure.getTracebackObject()))
    return None