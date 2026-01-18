import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_kv_python_init(self):
    from kivy.factory import Factory
    from kivy.lang import Builder
    from kivy.uix.widget import Widget

    class MyObject(object):
        value = 55

    class MyWidget(Widget):
        cheese = MyObject()
    Builder.load_string(dedent('\n        <MyWidget>:\n            x: 55\n            y: self.width + 10\n            height: self.cheese.value\n            width: 44\n\n        <MySecondWidget@Widget>:\n            x: 55\n            Widget:\n                x: 23\n        '))
    w = MyWidget(x=22, height=12, y=999)
    self.assertEqual(w.x, 22)
    self.assertEqual(w.width, 44)
    self.assertEqual(w.y, 44 + 10)
    self.assertEqual(w.height, 12)
    w2 = Factory.MySecondWidget(x=999)
    self.assertEqual(w2.x, 999)
    self.assertEqual(w2.children[0].x, 23)