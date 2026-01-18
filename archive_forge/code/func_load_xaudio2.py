import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
def load_xaudio2(dll_name):
    """This will attempt to load a version of XAudio2. Versions supported: 2.9, 2.8.
       While Windows 8 ships with 2.8 and Windows 10 ships with version 2.9, it is possible to install 2.9 on 8/8.1.
    """
    xaudio2 = dll_name
    if platform.architecture()[0] == '32bit':
        if platform.machine().endswith('64'):
            xaudio2 = os.path.join(os.environ['WINDIR'], 'SysWOW64', '{}.dll'.format(xaudio2))
    xaudio2_lib = ctypes.windll.LoadLibrary(xaudio2)
    x3d_lib = ctypes.cdll.LoadLibrary(xaudio2)
    return (xaudio2_lib, x3d_lib)