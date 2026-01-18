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
def test_containerWhereAllElementsAreKnown(self) -> None:
    """
        A L{crefutil._Container} where all of its elements are known at
        construction time is nonsensical and will result in errors in any call
        to addDependant.
        """
    container = crefutil._Container([1, 2, 3], list)
    self.assertRaises(AssertionError, container.addDependant, {}, 'ignore-me')