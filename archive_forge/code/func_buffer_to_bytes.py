import functools
import os
import sys
import re
import shutil
import types
from .encoding import DEFAULT_ENCODING
import platform
def buffer_to_bytes(buf):
    """Cast a buffer or memoryview object to bytes"""
    if isinstance(buf, memoryview):
        return buf.tobytes()
    if not isinstance(buf, bytes):
        return bytes(buf)
    return buf