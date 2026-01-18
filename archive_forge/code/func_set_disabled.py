from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
def set_disabled(self, value):
    value = bool(value)
    if value != self._disabled_value:
        self._disabled_value = value
        if value:
            self.inc_disabled()
        else:
            self.dec_disabled()
        return True