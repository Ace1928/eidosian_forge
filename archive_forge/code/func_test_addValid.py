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
def test_addValid(self):
    """Check that we can add valid set elements"""
    self.assertIs(self.instance.A.domain, Any)
    with self.assertRaises(AttributeError):
        self.instance.A.add(2)