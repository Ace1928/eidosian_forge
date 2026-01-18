from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_data_namespace(self):
    model = AbstractModel()
    model.a = Param()
    model.b = Param()
    model.c = Param()
    model.d = Param()
    model.A = Set()
    model.e = Param(model.A)
    md = DataPortal()
    md.load(model=model, filename=currdir + 'data16.dat')
    self.assertEqual(md.data(namespace='ns1'), {'a': {None: 1}, 'A': {None: [7, 9, 11]}, 'e': {9: 90, 7: 70, 11: 110}})
    self.assertEqual(md['ns1', 'a'], 1)
    self.assertEqual(sorted(md.namespaces(), key=lambda x: 'None' if x is None else x), [None, 'ns1', 'ns2', 'ns3', 'nsX'])
    self.assertEqual(sorted(md.keys()), ['A', 'a', 'b', 'c', 'd', 'e'])
    self.assertEqual(sorted(md.keys('ns1')), ['A', 'a', 'e'])
    self.assertEqual(sorted(md.values(), key=lambda x: tuple(sorted(x) + [0]) if type(x) is list else tuple(sorted(x.values())) if not type(x) is int else (x,)), [-4, -3, -2, -1, [1, 3, 5], {1: 10, 3: 30, 5: 50}])
    self.assertEqual(sorted(md.values('ns1'), key=lambda x: tuple(sorted(x) + [0]) if type(x) is list else tuple(sorted(x.values())) if not type(x) is int else (x,)), [1, [7, 9, 11], {7: 70, 9: 90, 11: 110}])
    self.assertEqual(sorted(md.items()), [('A', [1, 3, 5]), ('a', -1), ('b', -2), ('c', -3), ('d', -4), ('e', {1: 10, 3: 30, 5: 50})])
    self.assertEqual(sorted(md.items('ns1')), [('A', [7, 9, 11]), ('a', 1), ('e', {7: 70, 9: 90, 11: 110})])