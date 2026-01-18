import collections
import functools
import logging
import pycurl
import threading
import time
from io import BytesIO
from tornado import httputil
from tornado import ioloop
from tornado.escape import utf8, native_str
from tornado.httpclient import (
from tornado.log import app_log
from typing import Dict, Any, Callable, Union, Optional
import typing
def fetch_impl(self, request: HTTPRequest, callback: Callable[[HTTPResponse], None]) -> None:
    self._requests.append((request, callback, self.io_loop.time()))
    self._process_queue()
    self._set_timeout(0)