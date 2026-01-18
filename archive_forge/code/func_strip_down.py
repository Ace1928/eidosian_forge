from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
def strip_down(self, instance, touch):
    if not instance.collide_point(*touch.pos):
        return False
    touch.grab(self)
    self.dispatch('on_press')