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
def test_discardValid(self):
    """Check that we fail to remove an invalid set element without an exception"""
    with self.assertRaises(AttributeError):
        self.instance.A.discard(self.e2)