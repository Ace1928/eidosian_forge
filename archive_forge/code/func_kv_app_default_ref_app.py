from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def kv_app_default_ref_app():
    from kivy.app import App
    from kivy.lang import Builder

    class TestApp(UnitKivyApp, App):

        def build(self):
            return Builder.load_string(dedent('\n                Widget:\n                    obj: app.__self__\n                '))
    return TestApp()