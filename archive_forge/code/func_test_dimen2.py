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
def test_dimen2(self):
    try:
        self.model.A = Set(initialize=[1, 2, (3, 4)])
        self.instance = self.model.create_instance()
    except ValueError:
        pass
    else:
        self.fail('test_dimen2')
    self.model.A = Set(dimen=None, initialize=[1, 2, (3, 4)])
    self.instance = self.model.create_instance()