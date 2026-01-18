import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def dispatch_input(self):
    """Called by :meth:`EventLoopBase.idle()` to read events from input
        providers, pass events to postproc, and dispatch final events.
        """
    for provider in self.input_providers:
        provider.update(dispatch_fn=self._dispatch_input)
    for mod in self.postproc_modules:
        self.input_events = mod.process(events=self.input_events)
    input_events = self.input_events
    pop = input_events.pop
    post_dispatch_input = self.post_dispatch_input
    while input_events:
        post_dispatch_input(*pop(0))