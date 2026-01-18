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
def on_touch_down(self, touch):
    """Event called when a touch down event is initiated.

        .. versionchanged:: 1.9.0
            The touch `pos` is now transformed to window coordinates before
            this method is called. Before, the touch `pos` coordinate would be
            `(0, 0)` when this method was called.
        """
    for w in self.children[:]:
        if w.dispatch('on_touch_down', touch):
            return True