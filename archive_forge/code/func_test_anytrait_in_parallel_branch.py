import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_anytrait_in_parallel_branch(self):
    actual = parse('a:*,b')
    expected = trait('a', notify=False).anytrait() | trait('b')
    self.assertEqual(actual, expected)