import os
import subprocess
import sys
from debugpy import adapter, common
from debugpy.common import log, messaging, sockets
from debugpy.adapter import components, servers, sessions
@message_handler
def output_event(self, event):
    self.client.propagate_after_start(event)