from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
@property
def proxy_ref(self):
    """Return a proxy reference to the widget, i.e. without creating a
        reference to the widget. See `weakref.proxy
        <http://docs.python.org/2/library/weakref.html?highlight        =proxy#weakref.proxy>`_ for more information.

        .. versionadded:: 1.7.2
        """
    _proxy_ref = self._proxy_ref
    if _proxy_ref is not None:
        return _proxy_ref
    f = partial(_widget_destructor, self.uid)
    self._proxy_ref = _proxy_ref = WeakProxy(self, f)
    _widget_destructors[self.uid] = (f, _proxy_ref)
    return _proxy_ref