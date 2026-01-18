import inspect
import selectors
import socket
import threading
import time
from typing import Any, Callable, Optional, Union
from . import _logging
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import (
from ._url import parse_url
def setReconnect(reconnectInterval: int) -> None:
    global RECONNECT
    RECONNECT = reconnectInterval