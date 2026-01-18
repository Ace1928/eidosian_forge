import contextlib
import errno
import re
import sys
import typing
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from types import TracebackType
from typing import Dict, Set, Optional, Union, Iterator, IO, Iterable, TYPE_CHECKING, Type
@assignment_operator.setter
def assignment_operator(self, new_operator):
    if new_operator not in {'=', '?='}:
        raise ValueError('Operator must be one of: "=", or "?=" - got: ' + new_operator)
    self._assignment_operator = new_operator