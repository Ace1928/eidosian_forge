import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def ensure_window(self):
    """Ensure that we have a window.
        """
    import kivy.core.window
    if not self.window:
        Logger.critical('App: Unable to get a Window, abort.')
        sys.exit(1)