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
@functools.wraps(coro)
def post_coroutine(self, *args, **kwargs):
    try:
        return self.io_loop.run_sync(functools.partial(coro, self, *args, **kwargs), timeout=timeout)
    except TimeoutError as e:
        if self._test_generator is not None and getattr(self._test_generator, 'cr_running', True):
            self._test_generator.throw(e)
        raise