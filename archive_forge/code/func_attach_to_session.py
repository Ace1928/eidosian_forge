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
def attach_to_session(self, session):
    """Attaches this server to the specified Session as a Server component.

        Raises ValueError if the server already belongs to some session.
        """
    with _lock:
        if self.server is not None:
            raise ValueError
        log.info('Attaching {0} to {1}', self, session)
        self.server = Server(session, self)