from __future__ import annotations
import logging as py_logging
import sys
from inspect import getsourcefile
from io import BytesIO, TextIOWrapper
from logging import Formatter, LogRecord, StreamHandler, getLogger
from typing import List, Optional, Tuple
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.compat import currentframe
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._stdlib import STDLibLogObserver
def handlerAndBytesIO() -> tuple[StreamHandler[TextIOWrapper], BytesIO]:
    """
    Construct a 2-tuple of C{(StreamHandler, BytesIO)} for testing interaction
    with the 'logging' module.

    @return: handler and io object
    """
    output = BytesIO()
    template = py_logging.BASIC_FORMAT
    stream = TextIOWrapper(output, encoding='utf-8', newline='\n')
    formatter = Formatter(template)
    handler = StreamHandler(stream)
    handler.setFormatter(formatter)
    return (handler, output)