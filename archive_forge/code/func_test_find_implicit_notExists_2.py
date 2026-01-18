import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_find_implicit_notExists_2(self):
    cuid = ComponentUID('b:1,1.c.a:3')
    self.assertTrue(cuid.find_component_on(self.m) is None)