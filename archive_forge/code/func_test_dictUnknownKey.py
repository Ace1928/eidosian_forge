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
def test_dictUnknownKey(self) -> None:
    """
        L{crefutil._DictKeyAndValue} only support keys C{0} and C{1}.
        """
    d = crefutil._DictKeyAndValue({})
    self.assertRaises(RuntimeError, d.__setitem__, 2, 3)