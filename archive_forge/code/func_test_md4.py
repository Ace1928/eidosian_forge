from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md4(self):
    md = DataPortal()
    model = AbstractModel()
    model.A = Set()
    model.B = Set()
    model.C = Set()
    md.load(model=model, filename=currdir + 'data3.dat')
    self.assertEqual(set(md['A']), set([]))
    self.assertEqual(set(md['B']), set([(1, 2)]))
    self.assertEqual(set(md['C']), set([('a', 'b', 'c')]))