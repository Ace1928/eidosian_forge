import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_io7(self):
    OUTPUT = open('diffset.dat', 'w')
    OUTPUT.write('data;\n')
    OUTPUT.write('set B := 1;\n')
    OUTPUT.write('end;\n')
    OUTPUT.close()
    self.model.B = ContinuousSet(bounds=(0, 1))
    self.instance = self.model.create_instance('diffset.dat')
    self.assertEqual(len(self.instance.B), 2)