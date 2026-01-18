from io import StringIO
import sys
import logging
import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.util import IterationLogger, minIgnoreNone, maxIgnoreNone
from pyomo.common.log import LoggingIntercept
def test_IterationRecord(self):
    self.iterLogger.newIteration(self.iteration, self.thetak, self.objk, self.radius, self.stepNorm)
    self.assertEqual(len(self.iterLogger.iterations), 1)
    self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 5.0)