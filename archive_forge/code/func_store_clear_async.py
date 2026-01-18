from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_clear_async(self, callback):
    self.store_clear()
    callback(self)