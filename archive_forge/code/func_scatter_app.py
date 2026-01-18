from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def scatter_app():
    from kivy.app import App
    from kivy.uix.label import Label
    from kivy.uix.scatter import Scatter

    class TestApp(UnitKivyApp, App):

        def build(self):
            label = Label(text='Hello, World!', size=('200dp', '200dp'))
            scatter = Scatter(do_scale=False, do_rotation=False)
            scatter.add_widget(label)
            return scatter
    return TestApp()