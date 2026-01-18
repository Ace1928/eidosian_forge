import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_invalid_indentation2(self):
    Builder = self.import_builder()
    from kivy.lang import ParserException
    kv_code = '   BoxLayout:'
    try:
        Builder.load_string(kv_code)
        self.fail('Invalid indentation.')
    except ParserException as e:
        pass