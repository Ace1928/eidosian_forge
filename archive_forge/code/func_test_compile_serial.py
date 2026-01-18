import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_compile_serial(self):
    actual = compile_str('name1.name2')
    expected = [create_graph(NamedTraitObserver(name='name1', notify=True, optional=False), NamedTraitObserver(name='name2', notify=True, optional=False))]
    self.assertEqual(actual, expected)