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
def on_response(self, response_handler):
    """Registers a handler to invoke when a response is received for this request.
        The handler is invoked with Response as its sole argument.

        If response has already been received, invokes the handler immediately.

        It is guaranteed that self.response is set before the handler is invoked.
        If no response was received from the other party before the channel closed,
        self.response is a dummy Response with body=NoMoreMessages().

        The handler is always invoked asynchronously on an unspecified background
        thread - thus, the caller of on_response() can never be blocked or deadlocked
        by the handler.

        No further incoming messages are processed until the handler returns, except for
        responses to requests that have wait_for_response() invoked on them.
        """
    with self.channel:
        self._response_handlers.append(response_handler)
        self._enqueue_response_handlers()