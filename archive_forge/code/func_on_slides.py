from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def on_slides(self, *args):
    if self.slides:
        self.index = self.index % len(self.slides)
    self._insert_visible_slides()
    self._trigger_position_visible_slides()