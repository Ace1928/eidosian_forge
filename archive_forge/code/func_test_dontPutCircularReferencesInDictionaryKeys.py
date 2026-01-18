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
def test_dontPutCircularReferencesInDictionaryKeys(self) -> None:
    """
        If a dictionary key contains a circular reference (which is probably a
        bad practice anyway) it will be resolved by a
        L{crefutil._DictKeyAndValue}, not by placing a L{crefutil.NotKnown}
        into a dictionary key.
        """
    self.assertRaises(AssertionError, dict().__setitem__, crefutil.NotKnown(), 'value')