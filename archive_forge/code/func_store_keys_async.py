from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_keys_async(self, callback):
    try:
        keys = self.store_keys()
        callback(self, keys)
    except:
        callback(self, [])