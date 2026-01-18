import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_constructor_error(self):
    m = ConcreteModel()
    m.x = Var([1, 2])

    class Foo(object):
        pass
    self.assertRaisesRegex(TypeError, 'First argument to Reference constructors must be a component, component slice, Sequence, or Mapping \\(received Foo', Reference, Foo())
    self.assertRaisesRegex(TypeError, 'First argument to Reference constructors must be a component, component slice, Sequence, or Mapping \\(received int', Reference, 5)
    self.assertRaisesRegex(TypeError, 'First argument to Reference constructors must be a component, component slice, Sequence, or Mapping \\(received None', Reference, None)