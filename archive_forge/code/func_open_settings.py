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
def open_settings(self, *largs):
    """Open the application settings panel. It will be created the very
        first time, or recreated if the previously cached panel has been
        removed by :meth:`destroy_settings`. The settings panel will be
        displayed with the
        :meth:`display_settings` method, which by default adds the
        settings panel to the Window attached to your application. You
        should override that method if you want to display the
        settings panel differently.

        :return:
            True if the settings has been opened.

        """
    if self._app_settings is None:
        self._app_settings = self.create_settings()
    displayed = self.display_settings(self._app_settings)
    if displayed:
        return True
    return False