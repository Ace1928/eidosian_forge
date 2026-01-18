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
@property
def is_valid_file(self):
    """Returns true if the file is valid

        Invalid elements include error elements (Deb822ErrorElement) but also
        issues such as paragraphs with duplicate fields or "empty" files
        (a valid deb822 file contains at least one paragraph).
        """
    had_paragraph = False
    for paragraph in self:
        had_paragraph = True
        if not paragraph or paragraph.has_duplicate_fields:
            return False
    if not had_paragraph:
        return False
    return self.find_first_error_element() is None