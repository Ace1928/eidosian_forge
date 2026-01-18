import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
def test_filter_constructor(self):
    """Check that RangeSets can filter out unwanted elements"""

    def evenFilter(model, el):
        return el % 2 == 0
    self.instance.tmp = RangeSet(0, 10, filter=evenFilter)
    self.assertEqual(sorted([x for x in self.instance.tmp]), [0, 2, 4, 6, 8, 10])