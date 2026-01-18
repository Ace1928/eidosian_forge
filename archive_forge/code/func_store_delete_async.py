from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_delete_async(self, key, callback):
    try:
        value = self.delete(key)
        callback(self, key, value)
    except:
        callback(self, key, None)