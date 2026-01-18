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
def remove_widget(self, widget):
    """Remove a widget from a window
        """
    if widget not in self.children:
        return
    self.children.remove(widget)
    if widget.canvas in self.canvas.children:
        self.canvas.remove(widget.canvas)
    elif widget.canvas in self.canvas.after.children:
        self.canvas.after.remove(widget.canvas)
    elif widget.canvas in self.canvas.before.children:
        self.canvas.before.remove(widget.canvas)
    widget.parent = None
    widget.unbind(pos_hint=self._update_childsize, size_hint=self._update_childsize, size_hint_max=self._update_childsize, size_hint_min=self._update_childsize, size=self._update_childsize, pos=self._update_childsize)