from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def kv_app_ref_app():
    from kivy.app import App
    from kivy.lang import Builder
    from kivy.properties import ObjectProperty
    from kivy.uix.widget import Widget

    class MyWidget(Widget):
        obj = ObjectProperty(None)
    Builder.load_string(dedent('\n        <MyWidget>:\n            obj: app.__self__\n        '))

    class TestApp(UnitKivyApp, App):

        def build(self):
            return MyWidget()
    return TestApp()