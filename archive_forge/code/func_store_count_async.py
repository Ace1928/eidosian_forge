from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_count_async(self, callback):
    try:
        value = self.store_count()
        callback(self, value)
    except:
        callback(self, 0)