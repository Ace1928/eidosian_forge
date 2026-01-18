from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
def on_output(category, output):
    self.channel.send_event('output', {'category': category, 'output': output})