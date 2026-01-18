import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
def style_aware_write(fileobj: IO[str], msg: str) -> None:
    """
    Write a string to a fileobject and strip its ANSI style sequences if required by allow_style setting

    :param fileobj: the file object being written to
    :param msg: the string being written
    """
    if allow_style == AllowStyle.NEVER or (allow_style == AllowStyle.TERMINAL and (not fileobj.isatty())):
        msg = strip_style(msg)
    fileobj.write(msg)