from kivy.clock import Clock
from kivy.event import EventDispatcher
def async_exists(self, callback, key):
    """Asynchronous version of :meth:`exists`.

        :Callback arguments:
            `store`: :class:`AbstractStore` instance
                Store instance
            `key`: string
                Name of the key to search for
            `result`: boo
                Result of the query, None if any error
        """
    self._schedule(self.store_exists_async, key=key, callback=callback)