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
def sort_elements(self, *, key=None, reverse=False):
    """Sort the elements (abstract values) in this list.

        This method will sort the logical values of the list. It will
        attempt to preserve comments associated with a given value where
        possible.  Whether space and separators are preserved depends on
        the contents of the field as well as the formatting settings.

        Sorting (without reformatting) is likely to leave you with "awkward"
        whitespace. Therefore, you almost always want to apply reformatting
        such as the reformat_when_finished() method.

        Sorting will invalidate all ValueReferences.
        """
    comment_start_node = None
    vtype = self._vtype
    stype = self._stype

    def key_func(x):
        if key:
            return key(x[0])
        return x[0].convert_to_text()
    parts = []
    for node in self._token_list.iter_nodes():
        value = node.value
        if isinstance(value, Deb822Token) and value.is_comment:
            if comment_start_node is None:
                comment_start_node = node
            continue
        if isinstance(value, vtype):
            comments = []
            if comment_start_node is not None:
                for keep_node in comment_start_node.iter_next(skip_current=False):
                    if keep_node is node:
                        break
                    comments.append(keep_node.value)
            parts.append((value, comments))
            comment_start_node = None
    parts.sort(key=key_func, reverse=reverse)
    self._changed = True
    self._token_list.clear()
    first_value = True
    separator_is_space = self._default_separator_factory().is_whitespace
    for value, comments in parts:
        if first_value:
            first_value = False
            if comments:
                comments = [x for x in comments if not isinstance(x, stype)]
                self.append_newline()
        else:
            if not separator_is_space and (not any((isinstance(x, stype) for x in comments))):
                self.append_separator(space_after_separator=False)
            if comments:
                self.append_newline()
            else:
                self._token_list.append(Deb822WhitespaceToken(' '))
        self._token_list.extend(comments)
        self.append_value(value)