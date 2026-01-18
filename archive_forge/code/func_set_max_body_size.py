import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
def set_max_body_size(self, max_body_size: int) -> None:
    """Sets the body size limit for a single request.

        Overrides the value from `.HTTP1ConnectionParameters`.
        """
    self._max_body_size = max_body_size