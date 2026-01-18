import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
def test_mc_3d(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-2, 1), initialize=-1)
    m.y = Var(bounds=(-1, 2), initialize=0)
    m.e = Expression(expr=m.x * pow(exp(m.x) - m.y, 2))
    ccSurf, cvSurf, ccAffine, cvAffine = make3dPlot(m.e.expr, 30)
    self.assertAlmostEqual(ccSurf[48], 11.5655473482574)
    self.assertAlmostEqual(cvSurf[48], -15.28102124928224)
    self.assertAlmostEqual(ccAffine[48], 11.565547348257398)
    self.assertAlmostEqual(cvAffine[48], -23.131094696514797)