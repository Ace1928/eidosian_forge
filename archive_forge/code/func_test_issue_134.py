import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_issue_134(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2])
    m.J = Set(initialize=[4, 5])
    m.IJ = m.I * m.J
    self.assertEqual(len(m.IJ), 4)
    self.assertEqual(m.IJ.dimen, 2)
    m.IJ2 = m.IJ * m.IJ
    self.assertEqual(len(m.IJ2), 16)
    self.assertEqual(m.IJ2.dimen, 4)
    self.assertEqual(len(m.IJ), 4)
    self.assertEqual(m.IJ.dimen, 2)