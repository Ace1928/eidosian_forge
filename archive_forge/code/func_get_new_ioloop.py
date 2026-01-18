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
def get_new_ioloop(self) -> IOLoop:
    """Returns the `.IOLoop` to use for this test.

        By default, a new `.IOLoop` is created for each test.
        Subclasses may override this method to return
        `.IOLoop.current()` if it is not appropriate to use a new
        `.IOLoop` in each tests (for example, if there are global
        singletons using the default `.IOLoop`) or if a per-test event
        loop is being provided by another system (such as
        ``pytest-asyncio``).

        .. deprecated:: 6.3
           This method will be removed in Tornado 7.0.
        """
    return IOLoop(make_current=False)