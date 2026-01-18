import pytest
from string import ascii_letters
from random import randint
import gc
import sys
def test_kv_widget_creation(kivy_benchmark):
    from kivy.lang import Builder
    from kivy.uix.widget import Widget

    class MyWidget(Widget):
        pass
    Builder.load_string('\n<MyWidget>:\n    width: 55\n    height: 37\n    x: self.width + 5\n    y: self.height + 32\n')
    w = MyWidget()
    kivy_benchmark(MyWidget)