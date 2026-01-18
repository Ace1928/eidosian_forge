import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_loading_failed_1(self):
    Builder = self.import_builder()
    from kivy.lang import ParserException
    try:
        Builder.load_string(dedent('#:kivy 1.0\n            <TLangClass>:\n            '))
        self.fail('Invalid indentation.')
    except ParserException:
        pass