from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def assert_item_types(self, input_str, expected_types, strip_indent=6, config=None):
    """
    Run the lexer on the input string and assert that the result tokens match
    the expected
    """
    lines = [line[strip_indent:] for line in input_str.split('\n')]
    self.assertEqual(expected_types, [item.kind for item in markup.parse(lines, config)])