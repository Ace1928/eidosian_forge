import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def add_postproc_module(self, mod):
    """Add a postproc input module (DoubleTap, TripleTap, DeJitter
        RetainTouch are defaults)."""
    if mod not in self.postproc_modules:
        self.postproc_modules.append(mod)