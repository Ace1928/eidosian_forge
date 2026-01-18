import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.set_utils import (
def test_indexed_by(self):
    m = ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 10))
    m.space = ContinuousSet(bounds=(0, 10))
    m.set = Set(initialize=['a', 'b', 'c'])
    m.set2 = Set(initialize=[('a', 1), ('b', 2)])
    m.v = Var()
    m.v1 = Var(m.time)
    m.v2 = Var(m.time, m.space)
    m.v3 = Var(m.set, m.space, m.time)
    m.v4 = Var(m.time, m.set2)
    m.v5 = Var(m.set2, m.time, m.space)

    @m.Block()
    def b(b):
        b.v = Var()
        b.v1 = Var(m.time)
        b.v2 = Var(m.time, m.space)
        b.v3 = Var(m.set, m.space, m.time)

    @m.Block(m.time)
    def b1(b):
        b.v = Var()
        b.v1 = Var(m.space)
        b.v2 = Var(m.space, m.set)

    @m.Block(m.time, m.space)
    def b2(b):
        b.v = Var()
        b.v1 = Var(m.set)

        @b.Block()
        def b(bl):
            bl.v = Var()
            bl.v1 = Var(m.set)
            bl.v2 = Var(m.time)

    @m.Block(m.set2, m.time)
    def b3(b):
        b.v = Var()
        b.v1 = Var(m.space)

        @b.Block(m.space)
        def b(bb):
            bb.v = Var(m.set)
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, wrt=m.time, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
    disc.apply_to(m, wrt=m.space, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
    self.assertFalse(is_explicitly_indexed_by(m.v, m.time))
    self.assertTrue(is_explicitly_indexed_by(m.b.v2, m.space))
    self.assertTrue(is_explicitly_indexed_by(m.b.v3, m.time, m.space))
    self.assertFalse(is_in_block_indexed_by(m.v1, m.time))
    self.assertFalse(is_in_block_indexed_by(m.v2, m.set))
    self.assertTrue(is_in_block_indexed_by(m.b1[m.time[1]].v2, m.time))
    self.assertTrue(is_in_block_indexed_by(m.b2[m.time[1], m.space[1]].b.v1, m.time))
    self.assertTrue(is_in_block_indexed_by(m.b2[m.time[1], m.space[1]].b.v2, m.time))
    self.assertTrue(is_explicitly_indexed_by(m.b2[m.time[1], m.space[1]].b.v2, m.time))
    self.assertFalse(is_in_block_indexed_by(m.b2[m.time[1], m.space[1]].b.v1, m.set))
    self.assertFalse(is_in_block_indexed_by(m.b2[m.time[1], m.space[1]].b.v1, m.space, stop_at=m.b2[m.time[1], m.space[1]]))
    self.assertTrue(is_explicitly_indexed_by(m.v4, m.time, m.set2))
    self.assertTrue(is_explicitly_indexed_by(m.v5, m.time, m.set2, m.space))
    self.assertTrue(is_in_block_indexed_by(m.b3['a', 1, m.time[1]].v, m.set2))
    self.assertTrue(is_in_block_indexed_by(m.b3['a', 1, m.time[1]].v, m.time))
    self.assertTrue(is_in_block_indexed_by(m.b3['a', 1, m.time[1]].v1[m.space[1]], m.set2))
    self.assertFalse(is_in_block_indexed_by(m.b3['a', 1, m.time[1]].v1[m.space[1]], m.space))
    self.assertTrue(is_in_block_indexed_by(m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.set2))
    self.assertTrue(is_in_block_indexed_by(m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.time))
    self.assertTrue(is_in_block_indexed_by(m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.space))
    self.assertFalse(is_in_block_indexed_by(m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.set))
    self.assertFalse(is_in_block_indexed_by(m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.time, stop_at=m.b3['b', 2, m.time[2]]))
    self.assertFalse(is_in_block_indexed_by(m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.time, stop_at=m.b3))