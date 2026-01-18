from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
@message_handler
def debugpySystemInfo_request(self, request):
    result = {'debugpy': {'version': debugpy.__version__}}
    if self.server:
        try:
            pydevd_info = self.server.channel.request('pydevdSystemInfo')
        except Exception:
            pass
        else:
            result.update(pydevd_info)
    return result