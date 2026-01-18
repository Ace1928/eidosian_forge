import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
def finish_connect(trans, key, ov):
    ov.getresult()
    conn.setsockopt(socket.SOL_SOCKET, _overlapped.SO_UPDATE_CONNECT_CONTEXT, 0)
    return conn