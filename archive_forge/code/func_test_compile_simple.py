import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_compile_simple(self):
    actual = compile_str('name')
    expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False))]
    self.assertEqual(actual, expected)