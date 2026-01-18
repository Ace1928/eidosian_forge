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
def stop_serving():
    global listener
    try:
        if listener is not None:
            listener.close()
            listener = None
    except Exception:
        log.swallow_exception(level='warning')
    sessions.report_sockets()