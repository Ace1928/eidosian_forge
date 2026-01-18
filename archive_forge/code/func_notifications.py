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
def notifications(self) -> NotificationArea | None:
    from ..config import config
    if config.notifications and self.curdoc and self.curdoc.session_context and (self.curdoc not in self._notifications):
        from .notifications import NotificationArea
        js_events = {}
        if config.ready_notification:
            js_events['document_ready'] = {'type': 'success', 'message': config.ready_notification, 'duration': 3000}
        if config.disconnect_notification:
            js_events['connection_lost'] = {'type': 'error', 'message': config.disconnect_notification}
        self._notifications[self.curdoc] = notifications = NotificationArea(js_events=js_events)
        return notifications
    elif self.curdoc is None or self.curdoc.session_context is None:
        return self._notification
    else:
        return self._notifications.get(self.curdoc) if self.curdoc else None