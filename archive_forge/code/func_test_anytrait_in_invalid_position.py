import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_anytrait_in_invalid_position(self):
    invalid_expressions = ['*.*', '*:*', '*.name', '*.items', '*:name', '*.a,b', '[a.*,b].c']
    for expression in invalid_expressions:
        with self.subTest(expression=expression):
            with self.assertRaises(ValueError):
                parse(expression)