from functools import wraps
from kivy.context import Context
from kivy.base import ExceptionManagerBase
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
def on_context_created(self):
    """Override this method in order to load your kv file or do anything
        else with the newly created context.
        """
    pass