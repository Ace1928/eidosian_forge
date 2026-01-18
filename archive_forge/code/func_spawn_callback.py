import asyncio
import concurrent.futures
import datetime
import functools
import numbers
import os
import sys
import time
import math
import random
import warnings
from inspect import isawaitable
from tornado.concurrent import (
from tornado.log import app_log
from tornado.util import Configurable, TimeoutError, import_object
import typing
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable
def spawn_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
    """Calls the given callback on the next IOLoop iteration.

        As of Tornado 6.0, this method is equivalent to `add_callback`.

        .. versionadded:: 4.0
        """
    self.add_callback(callback, *args, **kwargs)