from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
@staticmethod
def inline_literals_using(cls: type) -> None:
    """
        Set class to be used for inclusion of string literals into a parser.

        Example::

            # default literal class used is Literal
            integer = Word(nums)
            date_str = integer("year") + '/' + integer("month") + '/' + integer("day")

            date_str.parse_string("1999/12/31")  # -> ['1999', '/', '12', '/', '31']


            # change to Suppress
            ParserElement.inline_literals_using(Suppress)
            date_str = integer("year") + '/' + integer("month") + '/' + integer("day")

            date_str.parse_string("1999/12/31")  # -> ['1999', '12', '31']
        """
    ParserElement._literalStringClass = cls