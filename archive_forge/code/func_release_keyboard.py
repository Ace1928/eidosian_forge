from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def release_keyboard(self, target=None):
    """.. versionadded:: 1.0.4

        Internal method for the widget to release the real-keyboard. Check
        :meth:`request_keyboard` to understand how it works.
        """
    if self.allow_vkeyboard:
        key = 'single' if self.single_vkeyboard else target
        if key not in self._keyboards:
            return
        keyboard = self._keyboards[key]
        callback = keyboard.callback
        if callback:
            keyboard.callback = None
            callback()
        keyboard.target = None
        self.remove_widget(keyboard.widget)
        if key != 'single' and key in self._keyboards:
            del self._keyboards[key]
    elif self._system_keyboard.callback:
        callback = self._system_keyboard.callback
        self._system_keyboard.callback = None
        callback()
        return True