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
def test_issue_148(self):
    legal = set(['a', 'b', 'c'])
    m = ConcreteModel()
    m.s = Set(initialize=['a', 'b'], within=legal)
    self.assertEqual(set(m.s), {'a', 'b'})
    with self.assertRaisesRegex(ValueError, 'Cannot add value d to Set s'):
        m.s.add('d')