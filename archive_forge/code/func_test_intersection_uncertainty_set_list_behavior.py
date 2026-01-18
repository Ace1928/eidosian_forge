import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
def test_intersection_uncertainty_set_list_behavior(self):
    """
        Test the 'all_sets' attribute of the IntersectionSet
        class behaves like a regular Python list.
        """
    iset = IntersectionSet(bset=BoxSet([[0, 2]]), aset=AxisAlignedEllipsoidalSet([0], [1]))
    all_sets = iset.all_sets
    all_sets.append(BoxSet([[1, 2]]))
    del all_sets[2:]
    all_sets.extend([BoxSet([[1, 2]]), EllipsoidalSet([0], [[1]], 2)])
    del all_sets[2:]
    all_sets[0]
    all_sets[1]
    all_sets[100:]
    all_sets[0:2:20]
    all_sets[0:2:1]
    all_sets[-20:-1:2]
    self.assertRaises(IndexError, lambda: all_sets[2])
    self.assertRaises(IndexError, lambda: all_sets[-3])
    with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
        all_sets[:] = all_sets[0]
    with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
        del all_sets[1]
    with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
        del all_sets[1:]
    with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
        del all_sets[:]
    with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
        all_sets.clear()
    with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
        all_sets[0:] = []
    with self.assertRaisesRegex(IndexError, 'assignment index out of range'):
        all_sets[-3] = BoxSet([[1, 1.5]])
    with self.assertRaisesRegex(IndexError, 'assignment index out of range'):
        all_sets[2] = BoxSet([[1, 1.5]])
    all_sets[3:] = [BoxSet([[1, 1.5]]), BoxSet([[1, 3]])]