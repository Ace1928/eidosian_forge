import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_FileDeterminism_to_SortComponents(self):
    self.assertEqual(FileDeterminism_to_SortComponents(FileDeterminism(0)), SortComponents.unsorted)
    self.assertEqual(FileDeterminism_to_SortComponents(FileDeterminism.ORDERED), SortComponents.deterministic)
    self.assertEqual(FileDeterminism_to_SortComponents(FileDeterminism.SORT_INDICES), SortComponents.indices)
    self.assertEqual(FileDeterminism_to_SortComponents(FileDeterminism.SORT_SYMBOLS), SortComponents.indices | SortComponents.alphabetical)