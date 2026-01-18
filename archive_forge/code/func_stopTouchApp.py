import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def stopTouchApp():
    """Stop the current application by leaving the main loop.

    See :mod:`kivy.app` for example usage.
    """
    if EventLoop is None:
        return
    if EventLoop.status in ('stopped', 'closed'):
        return
    if EventLoop.status != 'started':
        if not EventLoop.stopping:
            EventLoop.stopping = True
            Clock.schedule_once(lambda dt: stopTouchApp(), 0)
        return
    Logger.info('Base: Leaving application in progress...')
    EventLoop.close()