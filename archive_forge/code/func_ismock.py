import contextlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import MethodType, ModuleType
from typing import Any, Generator, Iterator, List, Optional, Sequence, Tuple, Union
from sphinx.util import logging
from sphinx.util.inspect import isboundmethod, safe_getattr
def ismock(subject: Any) -> bool:
    """Check if the object is mocked."""
    try:
        if safe_getattr(subject, '__sphinx_mock__', None) is None:
            return False
    except AttributeError:
        return False
    if isinstance(subject, _MockModule):
        return True
    if isinstance(subject, MethodType) and isboundmethod(subject):
        tmp_subject = subject.__func__
    else:
        tmp_subject = subject
    try:
        __mro__ = safe_getattr(type(tmp_subject), '__mro__', [])
        if len(__mro__) > 2 and __mro__[-2] is _MockObject:
            return True
    except AttributeError:
        pass
    return False