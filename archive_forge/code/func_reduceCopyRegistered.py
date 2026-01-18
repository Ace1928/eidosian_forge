from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def reduceCopyRegistered(cr: object) -> tuple[type[CopyRegisteredLoaded], tuple[()]]:
    """
    Externally implement C{__reduce__} for L{CopyRegistered}.

    @param cr: The L{CopyRegistered} instance.

    @return: a 2-tuple of callable and argument list, in this case
        L{CopyRegisteredLoaded} and no arguments.
    """
    return (CopyRegisteredLoaded, ())