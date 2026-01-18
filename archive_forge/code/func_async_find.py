from kivy.clock import Clock
from kivy.event import EventDispatcher
def async_find(self, callback, **filters):
    """Asynchronous version of :meth:`find`.

        The callback will be called for each entry in the result.

        :Callback arguments:
            `store`: :class:`AbstractStore` instance
                Store instance
            `key`: string
                Name of the key to search for, or None if we reach the end of
                the results
            `result`: bool
                Indicate True if the storage has been updated, or False if
                nothing has been done (no changes). None if any error.
        """
    self._schedule(self.store_find_async, callback=callback, filters=filters)