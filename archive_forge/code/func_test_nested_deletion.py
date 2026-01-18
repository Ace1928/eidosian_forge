import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_nested_deletion(self):
    m = self.m
    rd = _ReferenceDict(m.b[:, :].x[:, :])
    self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2 * 2 * 2)
    self.assertTrue((1, 5, 7, 10) in rd)
    del rd[1, 5, 7, 10]
    self.assertFalse((1, 5, 7, 10) in rd)
    self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2 * 2 * 2 - 1)
    rd = _ReferenceDict(m.b[:, 4].x[8, :])
    self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2)
    self.assertTrue((1, 10) in rd)
    del rd[1, 10]
    self.assertFalse((1, 10) in rd)
    self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2 - 1)