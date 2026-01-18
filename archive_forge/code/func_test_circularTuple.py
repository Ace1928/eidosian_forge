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
def test_circularTuple(self) -> None:
    """
        L{aot.jellyToAOT} can persist circular references through tuples.
        """
    l: _CircularTupleType = []
    t = (l, 4321)
    l.append(t)
    j1 = aot.jellyToAOT(l)
    oj = aot.unjellyFromAOT(j1)
    self.assertIsInstance(oj[0], tuple)
    self.assertIs(oj[0][0], oj)
    self.assertEqual(oj[0][1], 4321)