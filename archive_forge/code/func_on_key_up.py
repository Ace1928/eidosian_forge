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
def on_key_up(self, key, scancode=None, codepoint=None, modifier=None, **kwargs):
    """Event called when a key is released (same arguments as on_keyboard).
        """
    if 'unicode' in kwargs:
        Logger.warning('The use of the unicode parameter is deprecated, and will be removed in future versions. Use codepoint instead, which has identical semantics.')