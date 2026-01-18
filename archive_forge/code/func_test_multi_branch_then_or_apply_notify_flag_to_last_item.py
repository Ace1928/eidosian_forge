import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_multi_branch_then_or_apply_notify_flag_to_last_item(self):
    actual = parse('root.[a.b.c.d,value]:g')
    expected = trait('root').then(trait('a').trait('b').trait('c').trait('d', False) | trait('value', False)).trait('g')
    self.assertEqual(actual, expected)