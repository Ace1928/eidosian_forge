from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md11(self):
    cwd = os.getcwd()
    os.chdir(currdir)
    md = DataPortal()
    model = AbstractModel()
    model.A = Set()
    model.B = Set()
    model.C = Set()
    model.D = Set()
    md.load(model=model, filename=currdir + 'data11.dat')
    self.assertEqual(set(md['A']), set([]))
    self.assertEqual(set(md['B']), set([(1, 2)]))
    self.assertEqual(set(md['C']), set([('a', 'b', 'c')]))
    self.assertEqual(set(md['D']), set([1, 3, 5]))
    os.chdir(cwd)