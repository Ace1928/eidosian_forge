import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass
import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from yarl import URL
from . import hdrs
from .log import client_logger, internal_logger
def netrc_from_env() -> Optional[netrc.netrc]:
    """Load netrc from file.

    Attempt to load it from the path specified by the env-var
    NETRC or in the default location in the user's home directory.

    Returns None if it couldn't be found or fails to parse.
    """
    netrc_env = os.environ.get('NETRC')
    if netrc_env is not None:
        netrc_path = Path(netrc_env)
    else:
        try:
            home_dir = Path.home()
        except RuntimeError as e:
            client_logger.debug('Could not resolve home directory when trying to look for .netrc file: %s', e)
            return None
        netrc_path = home_dir / ('_netrc' if IS_WINDOWS else '.netrc')
    try:
        return netrc.netrc(str(netrc_path))
    except netrc.NetrcParseError as e:
        client_logger.warning('Could not parse .netrc file: %s', e)
    except OSError as e:
        netrc_exists = False
        with contextlib.suppress(OSError):
            netrc_exists = netrc_path.is_file()
        if netrc_env or netrc_exists:
            client_logger.warning('Could not read .netrc file: %s', e)
    return None