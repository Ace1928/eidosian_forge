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
def test_finite_setintersection(self):
    self._verify_finite_intersection(SetOf({1, 3, 2, 5}), SetOf({0, 2, 3, 4, 5}))
    self._verify_finite_intersection({1, 3, 2, 5}, SetOf({0, 2, 3, 4, 5}))
    self._verify_finite_intersection(SetOf({1, 3, 2, 5}), {0, 2, 3, 4, 5})
    self._verify_finite_intersection(RangeSet(ranges=(NR(-5, -1, 0), NR(2, 3, 0), NR(5, 5, 0), NR(10, 20, 0))), SetOf({0, 2, 3, 4, 5}))
    self._verify_finite_intersection(SetOf({1, 3, 2, 5}), RangeSet(ranges=(NR(2, 5, 0), NR(2, 5, 0), NR(6, 6, 0), NR(6, 6, 0), NR(6, 6, 0))))