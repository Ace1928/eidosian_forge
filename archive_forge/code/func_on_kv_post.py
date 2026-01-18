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
def on_kv_post(self, base_widget):
    pass