import asyncio
from collections.abc import Generator
import functools
import inspect
import logging
import os
import re
import signal
import socket
import sys
import unittest
import warnings
from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop, TimeoutError
from tornado import netutil
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.process import Subprocess
from tornado.log import app_log
from tornado.util import raise_exc_info, basestring_type
from tornado.web import Application
import typing
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine
from types import TracebackType
def gen_test(func: Optional[Callable[..., Union[Generator, 'Coroutine']]]=None, timeout: Optional[float]=None) -> Union[Callable[..., None], Callable[[Callable[..., Union[Generator, 'Coroutine']]], Callable[..., None]]]:
    """Testing equivalent of ``@gen.coroutine``, to be applied to test methods.

    ``@gen.coroutine`` cannot be used on tests because the `.IOLoop` is not
    already running.  ``@gen_test`` should be applied to test methods
    on subclasses of `AsyncTestCase`.

    Example::

        class MyTest(AsyncHTTPTestCase):
            @gen_test
            def test_something(self):
                response = yield self.http_client.fetch(self.get_url('/'))

    By default, ``@gen_test`` times out after 5 seconds. The timeout may be
    overridden globally with the ``ASYNC_TEST_TIMEOUT`` environment variable,
    or for each test with the ``timeout`` keyword argument::

        class MyTest(AsyncHTTPTestCase):
            @gen_test(timeout=10)
            def test_something_slow(self):
                response = yield self.http_client.fetch(self.get_url('/'))

    Note that ``@gen_test`` is incompatible with `AsyncTestCase.stop`,
    `AsyncTestCase.wait`, and `AsyncHTTPTestCase.fetch`. Use ``yield
    self.http_client.fetch(self.get_url())`` as shown above instead.

    .. versionadded:: 3.1
       The ``timeout`` argument and ``ASYNC_TEST_TIMEOUT`` environment
       variable.

    .. versionchanged:: 4.0
       The wrapper now passes along ``*args, **kwargs`` so it can be used
       on functions with arguments.

    """
    if timeout is None:
        timeout = get_async_test_timeout()

    def wrap(f: Callable[..., Union[Generator, 'Coroutine']]) -> Callable[..., None]:

        @functools.wraps(f)
        def pre_coroutine(self, *args, **kwargs):
            result = f(self, *args, **kwargs)
            if isinstance(result, Generator) or inspect.iscoroutine(result):
                self._test_generator = result
            else:
                self._test_generator = None
            return result
        if inspect.iscoroutinefunction(f):
            coro = pre_coroutine
        else:
            coro = gen.coroutine(pre_coroutine)

        @functools.wraps(coro)
        def post_coroutine(self, *args, **kwargs):
            try:
                return self.io_loop.run_sync(functools.partial(coro, self, *args, **kwargs), timeout=timeout)
            except TimeoutError as e:
                if self._test_generator is not None and getattr(self._test_generator, 'cr_running', True):
                    self._test_generator.throw(e)
                raise
        return post_coroutine
    if func is not None:
        return wrap(func)
    else:
        return wrap