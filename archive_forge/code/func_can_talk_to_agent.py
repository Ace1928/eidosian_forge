import array
import ctypes.wintypes
import platform
import struct
from paramiko.common import zero_byte
from paramiko.util import b
import _thread as thread
from . import _winapi
def can_talk_to_agent():
    """
    Check to see if there is a "Pageant" agent we can talk to.

    This checks both if we have the required libraries (win32all or ctypes)
    and if there is a Pageant currently running.
    """
    return bool(_get_pageant_window_object())