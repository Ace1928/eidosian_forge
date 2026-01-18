import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_bind_fstring(self):
    from kivy.lang import Builder
    label = Builder.load_string(dedent("\n        <TestLabel@Label>:\n            text: f'{self.pos}|{self.size}'\n        TestLabel:\n        "))
    assert label.text == '[0, 0]|[100, 100]'
    label.pos = (150, 200)
    assert label.text == '[150, 200]|[100, 100]'