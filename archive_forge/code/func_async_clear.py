from kivy.clock import Clock
from kivy.event import EventDispatcher
def async_clear(self, callback):
    """Asynchronous version of :meth:`clear`.
        """
    self._schedule(self.store_clear_async, callback=callback)