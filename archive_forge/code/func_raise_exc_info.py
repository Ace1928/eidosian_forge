import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def raise_exc_info(exc_info: Tuple[Optional[type], Optional[BaseException], Optional['TracebackType']]) -> typing.NoReturn:
    try:
        if exc_info[1] is not None:
            raise exc_info[1].with_traceback(exc_info[2])
        else:
            raise TypeError('raise_exc_info called with no exception')
    finally:
        exc_info = (None, None, None)