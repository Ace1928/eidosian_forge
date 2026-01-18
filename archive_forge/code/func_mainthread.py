from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
def mainthread(func):
    """Decorator that will schedule the call of the function for the next
    available frame in the mainthread. It can be useful when you use
    :class:`~kivy.network.urlrequest.UrlRequest` or when you do Thread
    programming: you cannot do any OpenGL-related work in a thread.

    Please note that this method will return directly and no result can be
    returned::

        @mainthread
        def callback(self, *args):
            print('The request succeeded!',
                  'This callback is called in the main thread.')


        self.req = UrlRequest(url='http://...', on_success=callback)

    .. versionadded:: 1.8.0
    """

    @wraps(func)
    def delayed_func(*args, **kwargs):

        def callback_func(dt):
            func(*args, **kwargs)
        Clock.schedule_once(callback_func, 0)
    return delayed_func