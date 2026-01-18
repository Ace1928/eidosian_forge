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
def test_get_interval(self):
    self.assertEqual(Any.get_interval(), (None, None, None))
    a = UnindexedComponent_set
    self.assertEqual(a.get_interval(), (None, None, None))
    a = Set(initialize=['a'])
    a.construct()
    self.assertEqual(a.get_interval(), ('a', 'a', None))
    a = Set(initialize=[1])
    a.construct()
    self.assertEqual(a.get_interval(), (1, 1, 0))