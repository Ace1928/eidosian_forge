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
def propagate(self, message):
    """Sends a new message with the same type and payload.

        If it was a request, returns the new OutgoingRequest object for it.
        """
    assert message.is_request() or message.is_event()
    if message.is_request():
        return self.send_request(message.command, message.arguments)
    else:
        self.send_event(message.event, message.body)