from __future__ import annotations
import argparse
import os
import signal
import sys
import threading
from .sync.client import ClientConnection, connect
from .version import version as websockets_version
def print_incoming_messages(websocket: ClientConnection, stop: threading.Event) -> None:
    for message in websocket:
        if isinstance(message, str):
            print_during_input('< ' + message)
        else:
            print_during_input('< (binary) ' + message.hex())
    if not stop.is_set():
        if sys.platform == 'win32':
            ctrl_c = signal.CTRL_C_EVENT
        else:
            ctrl_c = signal.SIGINT
        os.kill(os.getpid(), ctrl_c)