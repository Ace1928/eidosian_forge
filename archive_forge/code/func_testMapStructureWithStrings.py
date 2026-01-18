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
def testMapStructureWithStrings(self):
    ab_tuple = collections.namedtuple('ab_tuple', 'a, b')
    inp_a = ab_tuple(a='foo', b=('bar', 'baz'))
    inp_b = ab_tuple(a=2, b=(1, 3))
    out = tree.map_structure(lambda string, repeats: string * repeats, inp_a, inp_b)
    self.assertEqual('foofoo', out.a)
    self.assertEqual('bar', out.b[0])
    self.assertEqual('bazbazbaz', out.b[1])
    nt = ab_tuple(a=('something', 'something_else'), b='yet another thing')
    rev_nt = tree.map_structure(lambda x: x[::-1], nt)
    tree.assert_same_structure(nt, rev_nt)
    self.assertEqual(nt.a[0][::-1], rev_nt.a[0])
    self.assertEqual(nt.a[1][::-1], rev_nt.a[1])
    self.assertEqual(nt.b[::-1], rev_nt.b)