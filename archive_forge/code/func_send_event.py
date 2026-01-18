from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
def send_event(self, event, body=None):
    """Sends a new event.

        If body is None or {}, "body" will be omitted in JSON.

        Safe to call concurrently for the same channel from different threads.
        """
    d = {'type': 'event', 'event': event}
    if body is not None and body != {}:
        d['body'] = body
    with self._send_message(d):
        pass