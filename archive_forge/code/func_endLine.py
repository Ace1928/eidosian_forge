from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
def endLine(self, message, color):
    """
        Print 'message' in the given color.

        @param message: A string message, usually '[OK]' or something similar.
        @param color: A string color, 'red', 'green' and so forth.
        """
    spaces = ' ' * (self.columns - len(self.currentLine) - len(message))
    super()._write(spaces)
    self._colorizer.write(message, color)
    super()._write('\n')