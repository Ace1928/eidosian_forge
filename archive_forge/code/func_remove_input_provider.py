import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def remove_input_provider(self, provider):
    """Remove an input provider.

        .. versionchanged:: 2.1.0
            Provider will be also removed if it exist in auto-remove list.
        """
    if provider in self.input_providers:
        self.input_providers.remove(provider)
        if provider in self.input_providers_autoremove:
            self.input_providers_autoremove.remove(provider)