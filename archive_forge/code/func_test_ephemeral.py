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
def test_ephemeral(self) -> None:
    o = MyEphemeral(3)
    self.assertEqual(o.__class__, MyEphemeral)
    self.assertEqual(o.x, 3)
    pickl = pickle.dumps(o)
    o = pickle.loads(pickl)
    self.assertEqual(o.__class__, styles.Ephemeral)
    self.assertFalse(hasattr(o, 'x'))