import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def find_first_error_element(self):
    """Returns the first Deb822ErrorElement (or None) in the file"""
    return next(iter(self.iter_recurse(only_element_or_token_type=Deb822ErrorElement)), None)