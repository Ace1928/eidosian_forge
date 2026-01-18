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
@property
def as_substvar(self):
    """Provides a mapping to the Substvars object for more advanced operations

        Treating a substvars file mostly as a "str -> str" mapping is sufficient for many cases.
        But when full control over the substvars (like fiddling with the assignment operator) is
        needed this attribute is useful.

        >>> content = '''
        ... # Some comment (which is allowed but no one uses them - also, they are not preserved)
        ... shlib:Depends=foo (>= 1.0), libbar2 (>= 2.1-3~)
        ... random:substvar?=With the new assignment operator from dpkg 1.21.8
        ... '''
        >>> substvars = Substvars()
        >>> substvars.read_substvars(content.splitlines())
        >>> substvars.as_substvar["shlib:Depends"].assignment_operator
        '='
        >>> substvars.as_substvar["random:substvar"].assignment_operator
        '?='
        >>> # Mutation is also possible
        >>> substvars.as_substvar["shlib:Depends"].assignment_operator = '?='
        >>> print(substvars.dump(), end="")
        shlib:Depends?=foo (>= 1.0), libbar2 (>= 2.1-3~)
        random:substvar?=With the new assignment operator from dpkg 1.21.8
        """
    return self._vars