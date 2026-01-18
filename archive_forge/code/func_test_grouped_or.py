import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_grouped_or(self):
    actual = parse('root.[left,right]')
    expected = trait('root').then(trait('left') | trait('right'))
    self.assertEqual(actual, expected)