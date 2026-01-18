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
def test_copyReg(self) -> None:
    """
        L{aot.jellyToSource} and L{aot.unjellyFromSource} honor functions
        registered in the pickle copy registry.
        """
    uj = aot.unjellyFromSource(aot.jellyToSource(CopyRegistered()))
    self.assertIsInstance(uj, CopyRegisteredLoaded)