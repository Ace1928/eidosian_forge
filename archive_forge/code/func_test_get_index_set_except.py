import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.set_utils import (
def test_get_index_set_except(self):
    """
        Tests:
          For components indexed by 0, 1, 2, 3, 4 sets:
            get_index_set_except one, then two (if any) of those sets
            check two items that should be in set_except
            insert item(s) back into these sets via index_getter
        """
    m = ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 10))
    m.space = ContinuousSet(bounds=(0, 10))
    m.set1 = Set(initialize=['a', 'b', 'c'])
    m.set2 = Set(initialize=['d', 'e', 'f'])
    m.v = Var()
    m.v1 = Var(m.time)
    m.v2 = Var(m.time, m.space)
    m.v3 = Var(m.time, m.space, m.set1)
    m.v4 = Var(m.time, m.space, m.set1, m.set2)
    m.set3 = Set(initialize=[('a', 1), ('b', 2)])
    m.v5 = Var(m.set3)
    m.v6 = Var(m.time, m.space, m.set3)
    m.v7 = Var(m.set3, m.space, m.time)
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, wrt=m.time, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
    disc.apply_to(m, wrt=m.space, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
    info = get_index_set_except(m.v1, m.time)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertTrue(set_except == [None])
    self.assertEqual(index_getter((), -1), -1)
    info = get_index_set_except(m.v2, m.time)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertTrue(m.space[1] in set_except and m.space.last() in set_except)
    self.assertEqual(index_getter((2,), 4), (4, 2))
    self.assertEqual(index_getter(2, 4), (4, 2))
    info = get_index_set_except(m.v2, m.space, m.time)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertTrue(set_except == [None])
    self.assertEqual(index_getter((), 5, 7), (7, 5))
    info = get_index_set_except(m.v3, m.time)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertTrue((m.space[1], 'b') in set_except and (m.space.last(), 'a') in set_except)
    self.assertEqual(index_getter((2, 'b'), 7), (7, 2, 'b'))
    info = get_index_set_except(m.v3, m.space, m.time)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertTrue('a' in set_except)
    self.assertEqual(index_getter('b', 1.2, 1.1), (1.1, 1.2, 'b'))
    info = get_index_set_except(m.v4, m.set1, m.space)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertTrue((m.time[1], 'd') in set_except)
    self.assertEqual(index_getter((4, 'f'), 'b', 8), (4, 8, 'b', 'f'))
    index_set = m.v4.index_set()
    for partial_index in set_except:
        complete_index = index_getter(partial_index, 'a', m.space[2])
        self.assertTrue(complete_index in index_set)
    info = get_index_set_except(m.v5, m.set3)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertEqual(set_except, [None])
    self.assertEqual(index_getter((), ('a', 1)), ('a', 1))
    info = get_index_set_except(m.v6, m.set3, m.time)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertTrue(m.space[1] in set_except)
    self.assertEqual(index_getter(m.space[1], ('b', 2), m.time[1]), (m.time[1], m.space[1], 'b', 2))
    info = get_index_set_except(m.v7, m.time)
    set_except = info['set_except']
    index_getter = info['index_getter']
    self.assertIn(('a', 1, m.space[1]), set_except)
    self.assertEqual(index_getter(('a', 1, m.space[1]), m.time[1]), ('a', 1, m.space[1], m.time[1]))
    m.v8 = Var(m.time, m.set3, m.time)
    with self.assertRaises(ValueError):
        info = get_index_set_except(m.v8, m.time)
    with self.assertRaises(ValueError):
        info = get_index_set_except(m.v8, m.space)