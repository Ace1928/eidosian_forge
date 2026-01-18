import os
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.common.gsl import find_GSL
from pyomo.environ import ConcreteModel, ExternalFunction, Var, Objective
def assertListsAlmostEqual(self, first, second, places=7, msg=None):
    self.assertEqual(len(first), len(second))
    msg = 'lists %s and %s differ at item ' % (first, second)
    for i, a in enumerate(first):
        self.assertAlmostEqual(a, second[i], places, msg + str(i))