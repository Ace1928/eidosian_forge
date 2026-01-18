from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
def propagate_after_start(self, event):
    if self._deferred_events is not None:
        self._deferred_events.append(event)
        log.debug('Propagation deferred.')
    else:
        self.client.channel.propagate(event)