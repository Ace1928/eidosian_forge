from kivy.core.text import DEFAULT_FONT
from kivy.uix.modalview import ModalView
from kivy.properties import (StringProperty, ObjectProperty, OptionProperty,
def on_content(self, instance, value):
    if self._container:
        self._container.clear_widgets()
        self._container.add_widget(value)