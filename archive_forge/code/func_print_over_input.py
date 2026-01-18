from __future__ import annotations
import argparse
import os
import signal
import sys
import threading
from .sync.client import ClientConnection, connect
from .version import version as websockets_version
def print_over_input(string: str) -> None:
    sys.stdout.write(f'\r\x1b[K{string}\n')
    sys.stdout.flush()