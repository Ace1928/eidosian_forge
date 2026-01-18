from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
import debugpy
from debugpy import adapter
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import components, sessions
import traceback
import io
@message_handler
def exited_event(self, event: messaging.Event):
    if event('pydevdReason', str, optional=True) == 'processReplaced':
        self.connection.process_replaced = True
    elif not self.launcher:
        self.client.propagate_after_start(event)