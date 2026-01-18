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
def split_fd(self, fd: Union[int, _Selectable]) -> Tuple[int, Union[int, _Selectable]]:
    if isinstance(fd, int):
        return (fd, fd)
    return (fd.fileno(), fd)