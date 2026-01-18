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
def read_substvars(self, fileobj):
    """Read substvars from an open text file in the format supported by dpkg-gencontrol

        On success, all existing variables will be discarded and only variables
        from the file will be present after this method completes.  In case of
        any IO related errors, the object retains its state prior to the call
        of this method.

        >>> content = '''
        ... # Some comment (which is allowed but no one uses them - also, they are not preserved)
        ... shlib:Depends=foo (>= 1.0), libbar2 (>= 2.1-3~)
        ... random:substvar?=With the new assignment operator from dpkg 1.21.8
        ... '''
        >>> substvars = Substvars()
        >>> substvars.read_substvars(content.splitlines())
        >>> substvars["shlib:Depends"]
        'foo (>= 1.0), libbar2 (>= 2.1-3~)'
        >>> substvars["random:substvar"]
        'With the new assignment operator from dpkg 1.21.8'

        :param fileobj: An open file (in text mode using the UTF-8 encoding) or an
          iterable of str that provides line by line content.
        """
    vars_dict = OrderedDict()
    for line in fileobj:
        if line.strip() == '' or line[0] == '#':
            continue
        m = _SUBSTVAR_PATTERN.match(line.rstrip('\r\n'))
        if not m:
            continue
        varname, assignment_operator, value = m.groups()
        vars_dict[varname] = Substvar(value, assignment_operator=assignment_operator)
    self._vars = vars_dict