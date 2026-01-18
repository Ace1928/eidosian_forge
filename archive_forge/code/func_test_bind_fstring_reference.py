import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_bind_fstring_reference(self):
    from kivy.lang import Builder
    root = Builder.load_string(dedent("\n        FloatLayout:\n            Label:\n                id: original\n                text: 'perfect'\n            Label:\n                id: duplicate\n                text: f'{original.text}'\n        "))
    assert root.ids.duplicate.text == 'perfect'
    root.ids.original.text = 'new text'
    assert root.ids.duplicate.text == 'new text'