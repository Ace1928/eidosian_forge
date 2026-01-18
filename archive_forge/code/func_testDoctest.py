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
def testDoctest(self):
    extraglobs = {'collections': collections, 'tree': tree}
    num_failed, num_attempted = doctest.testmod(tree, extraglobs=extraglobs, optionflags=doctest.ELLIPSIS)
    self.assertGreater(num_attempted, 0, 'No doctests found.')
    self.assertEqual(num_failed, 0, '{} doctests failed'.format(num_failed))