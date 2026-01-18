from __future__ import annotations
import asyncio
import datetime as dt
import inspect
import logging
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import (
from urllib.parse import urljoin
from weakref import WeakKeyDictionary
import param
from bokeh.document import Document
from bokeh.document.locking import UnlockedDocumentProxy
from bokeh.io import curdoc as _curdoc
from pyviz_comms import CommManager as _CommManager
from ..util import decode_token, parse_timedelta
from .logging import LOG_SESSION_RENDERED, LOG_USER_MSG
@property
def browser_info(self) -> BrowserInfo | None:
    from ..config import config
    from .browser import BrowserInfo
    if config.browser_info and self.curdoc and self.curdoc.session_context and (self.curdoc not in self._browsers):
        browser = self._browsers[self.curdoc] = BrowserInfo()
    elif self.curdoc is None:
        if self._browser is None and config.browser_info:
            self._browser = BrowserInfo()
        browser = self._browser
    else:
        browser = self._browsers.get(self.curdoc) if self.curdoc else None
    return browser