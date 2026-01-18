import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_io3(self):
    OUTPUT = open('diffset.dat', 'w')
    OUTPUT.write('data;\n')
    OUTPUT.write('set A := 1 3 5;\n')
    OUTPUT.write('end;\n')
    OUTPUT.close()
    self.model.A = ContinuousSet(bounds=(2, 6))
    with self.assertRaisesRegex(ValueError, 'The value is not in the domain \\[2..6\\]'):
        self.instance = self.model.create_instance('diffset.dat')