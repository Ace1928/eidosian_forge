from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def on_load_func(self, instance, value):
    if value:
        Clock.schedule_once(self._do_initial_load)