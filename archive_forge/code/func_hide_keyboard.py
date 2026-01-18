from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def hide_keyboard(self):
    """
        Convenience function to hide the keyboard in managed mode.
        """
    if self.keyboard_mode == 'managed':
        self._unbind_keyboard()