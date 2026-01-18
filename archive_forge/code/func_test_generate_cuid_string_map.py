import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_generate_cuid_string_map(self):
    model = Block(concrete=True)
    model.x = Var()
    model.y = Var([1, 2])
    model.V = Var([('a', 'b'), (1, '2'), (3, 4)])
    model.b = Block(concrete=True)
    model.b.z = Var([1, '2'])
    setattr(model.b, '.H', Var(['a', 2]))
    model.B = Block(['a', 2], concrete=True)
    setattr(model.B['a'], '.k', Var())
    model.B[2].b = Block()
    model.B[2].b.x = Var()
    model.add_component('c tuple', Constraint(Any))
    model.component('c tuple')[1,] = model.x >= 0
    cuids = (ComponentUID.generate_cuid_string_map(model, repr_version=1), ComponentUID.generate_cuid_string_map(model))
    self.assertEqual(len(cuids[0]), 24)
    self.assertEqual(len(cuids[1]), 24)
    for obj in [model, model.x, model.y, model.y[1], model.y[2], model.V, model.V['a', 'b'], model.V[1, '2'], model.V[3, 4], model.b, model.b.z, model.b.z[1], model.b.z['2'], getattr(model.b, '.H'), getattr(model.b, '.H')['a'], getattr(model.b, '.H')[2], model.B, model.B['a'], getattr(model.B['a'], '.k'), model.B[2], model.B[2].b, model.B[2].b.x, model.component('c tuple')[1,]]:
        self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
        self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])
    cuids = (ComponentUID.generate_cuid_string_map(model, descend_into=False, repr_version=1), ComponentUID.generate_cuid_string_map(model, descend_into=False))
    self.assertEqual(len(cuids[0]), 15)
    self.assertEqual(len(cuids[1]), 15)
    for obj in [model, model.x, model.y, model.y[1], model.y[2], model.V, model.V['a', 'b'], model.V[1, '2'], model.V[3, 4], model.b, model.B, model.B['a'], model.B[2], model.component('c tuple')[1,]]:
        self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
        self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])
    cuids = (ComponentUID.generate_cuid_string_map(model, ctype=Var, repr_version=1), ComponentUID.generate_cuid_string_map(model, ctype=Var))
    self.assertEqual(len(cuids[0]), 22)
    self.assertEqual(len(cuids[1]), 22)
    for obj in [model, model.x, model.y, model.y[1], model.y[2], model.V, model.V['a', 'b'], model.V[1, '2'], model.V[3, 4], model.b, model.b.z, model.b.z[1], model.b.z['2'], getattr(model.b, '.H'), getattr(model.b, '.H')['a'], getattr(model.b, '.H')[2], model.B, model.B['a'], getattr(model.B['a'], '.k'), model.B[2], model.B[2].b, model.B[2].b.x]:
        self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
        self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])
    cuids = (ComponentUID.generate_cuid_string_map(model, ctype=Var, descend_into=False, repr_version=1), ComponentUID.generate_cuid_string_map(model, ctype=Var, descend_into=False))
    self.assertEqual(len(cuids[0]), 9)
    self.assertEqual(len(cuids[1]), 9)
    for obj in [model, model.x, model.y, model.y[1], model.y[2], model.V, model.V['a', 'b'], model.V[1, '2'], model.V[3, 4]]:
        self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
        self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])