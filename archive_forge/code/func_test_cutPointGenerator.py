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
def test_cutPointGenerator(self):
    CG = SetProduct_InfiniteSet._cutPointGenerator
    i = Any
    j = SetOf([(1, 1), (1, 2), (2, 1), (2, 2)])
    test = list((tuple(_) for _ in CG((i, i), 3)))
    ref = [(0, 0, 3), (0, 1, 3), (0, 2, 3), (0, 3, 3)]
    self.assertEqual(test, ref)
    test = list((tuple(_) for _ in CG((i, i, i), 3)))
    ref = [(0, 0, 0, 3), (0, 0, 1, 3), (0, 0, 2, 3), (0, 0, 3, 3), (0, 1, 1, 3), (0, 1, 2, 3), (0, 1, 3, 3), (0, 2, 2, 3), (0, 2, 3, 3), (0, 3, 3, 3)]
    self.assertEqual(test, ref)
    test = list((tuple(_) for _ in CG((i, j, i), 5)))
    ref = [(0, 0, 2, 5), (0, 1, 3, 5), (0, 2, 4, 5), (0, 3, 5, 5)]
    self.assertEqual(test, ref)