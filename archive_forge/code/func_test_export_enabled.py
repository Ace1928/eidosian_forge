import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def test_export_enabled(self):
    model = ConcreteModel()
    model.test_implicit = Suffix()
    self.assertFalse(model.test_implicit.export_enabled())
    model.test_local = Suffix(direction=Suffix.LOCAL)
    self.assertFalse(model.test_local.export_enabled())
    model.test_out = Suffix(direction=Suffix.IMPORT)
    self.assertFalse(model.test_out.export_enabled())
    model.test_in = Suffix(direction=Suffix.EXPORT)
    self.assertTrue(model.test_in.export_enabled())
    model.test_inout = Suffix(direction=Suffix.IMPORT_EXPORT)
    self.assertTrue(model.test_inout.export_enabled())