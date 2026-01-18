import abc
import collections
import threading
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_messaging._drivers import common
def iter_free(self):
    """Iterate over free items."""
    while True:
        try:
            _, item = self._items.pop()
            yield item
        except IndexError:
            return