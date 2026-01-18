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
def add_widget(self, widget, canvas=None):
    """Add a widget to a window"""
    if widget.parent:
        from kivy.uix.widget import WidgetException
        raise WidgetException('Cannot add %r to window, it already has a parent %r' % (widget, widget.parent))
    widget.parent = self
    self.children.insert(0, widget)
    canvas = self.canvas.before if canvas == 'before' else self.canvas.after if canvas == 'after' else self.canvas
    canvas.add(widget.canvas)
    self.update_childsize([widget])
    widget.bind(pos_hint=self._update_childsize, size_hint=self._update_childsize, size_hint_max=self._update_childsize, size_hint_min=self._update_childsize, size=self._update_childsize, pos=self._update_childsize)