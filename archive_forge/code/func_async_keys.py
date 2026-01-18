from kivy.clock import Clock
from kivy.event import EventDispatcher
def async_keys(self, callback):
    """Asynchronously return all the keys in the storage.
        """
    self._schedule(self.store_keys_async, callback=callback)