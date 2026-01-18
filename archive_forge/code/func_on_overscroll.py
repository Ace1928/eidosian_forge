from kivy.effects.scroll import ScrollEffect
from kivy.properties import NumericProperty, BooleanProperty
from kivy.metrics import sp
def on_overscroll(self, *args):
    self.trigger_velocity_update()