import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
def testUnflattenSequenceAs_wrongLengthsError(self):
    with self.assertRaisesRegex(ValueError, 'Structure had 2 elements, but flat_sequence had 3 elements.'):
        tree.unflatten_as(['hello', 'world'], ['and', 'goodbye', 'again'])