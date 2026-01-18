import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
def get_system_locale() -> str:
    """Retrieve the string representing the system locale."""
    local_name = create_unicode_buffer(LOCALE_NAME_MAX_LENGTH)
    kernel32.GetUserDefaultLocaleName(local_name, LOCALE_NAME_MAX_LENGTH)
    return local_name.value