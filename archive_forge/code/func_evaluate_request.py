from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
@message_handler
def evaluate_request(self, request):
    propagated_request = self.server.channel.propagate(request)

    def handle_response(response):
        request.respond(response.body)
    propagated_request.on_response(handle_response)
    return messaging.NO_RESPONSE