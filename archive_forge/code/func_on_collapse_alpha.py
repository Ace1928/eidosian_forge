from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (ObjectProperty, StringProperty,
from kivy.uix.widget import Widget
from kivy.logger import Logger
def on_collapse_alpha(self, instance, value):
    self.accordion._trigger_layout()