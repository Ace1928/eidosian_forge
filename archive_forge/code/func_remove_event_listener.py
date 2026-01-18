import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def remove_event_listener(self, listener):
    """Remove an event listener from the list.
        """
    if listener in self.event_listeners:
        self.event_listeners.remove(listener)