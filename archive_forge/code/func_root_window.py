from appearing. If you want to prevent the settings instance from appearing
from the same thread and the other coroutines are only executed when Kivy
import os
from inspect import getfile
from os.path import dirname, join, exists, sep, expanduser, isfile
from kivy.config import ConfigParser
from kivy.base import runTouchApp, async_runTouchApp, stopTouchApp
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty
from kivy.setupconfig import USE_SDL2
@property
def root_window(self):
    """.. versionadded:: 1.9.0

        Returns the root window instance used by :meth:`run`.
        """
    return self._app_window