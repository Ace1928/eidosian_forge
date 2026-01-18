import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def errno_from_exception(e: BaseException) -> Optional[int]:
    """Provides the errno from an Exception object.

    There are cases that the errno attribute was not set so we pull
    the errno out of the args but if someone instantiates an Exception
    without any args you will get a tuple error. So this function
    abstracts all that behavior to give you a safe way to get the
    errno.
    """
    if hasattr(e, 'errno'):
        return e.errno
    elif e.args:
        return e.args[0]
    else:
        return None