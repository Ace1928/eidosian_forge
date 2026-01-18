import inspect
from threading import Thread, Event
from kivy.app import App
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.utils import deprecated
def safeIn(self):
    """Provides a thread-safe entry point for interactive launching."""
    self.safe.clear()
    Clock.schedule_once(safeWait, -1)
    self.confirmed.wait()