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
def test_basicIdentity(self) -> None:
    aj = aot.AOTJellier().jellyToAO
    d = {'hello': 'world', 'method': aj}
    l = [1, 2, 3, 'he\tllo\n\n"x world!', 'goodbye \n\tတ world!', 1, 1.0, 100 ** 100, unittest, aot.AOTJellier, d, funktion]
    t = tuple(l)
    l.append(l)
    l.append(t)
    l.append(t)
    uj = aot.unjellyFromSource(aot.jellyToSource([l, l]))
    assert uj[0] is uj[1]
    assert uj[1][0:5] == l[0:5]