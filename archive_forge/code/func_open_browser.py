from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def open_browser(url: str) -> None:
    """Open a web browser pointing to a given URL.

    We use this function instead of Python's `webbrowser` module because this
    way we can capture stdout/stderr to avoid polluting the terminal with the
    browser's messages. For example, Chrome always prints things like "Created
    new window in existing browser session", and those get on the user's way.

    url : str
        The URL. Must include the protocol.

    """
    if env_util.IS_WINDOWS:
        _open_browser_with_webbrowser(url)
        return
    if env_util.IS_LINUX_OR_BSD:
        if env_util.is_executable_in_path('xdg-open'):
            _open_browser_with_command('xdg-open', url)
            return
        _open_browser_with_webbrowser(url)
        return
    if env_util.IS_DARWIN:
        _open_browser_with_command('open', url)
        return
    import platform
    raise Error('Cannot open browser in platform "%s"' % platform.system())