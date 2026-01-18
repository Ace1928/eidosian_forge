from collections import deque
import ctypes
import threading
from typing import Deque, Optional, TYPE_CHECKING
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
from . import lib_pulseaudio as pa
from .interface import PulseAudioMainloop
def dump_debug_info(self):
    print('Client version: ', pa.pa_get_library_version())
    print('Server:         ', self.context.server)
    print('Protocol:       ', self.context.protocol_version)
    print('Server protocol:', self.context.server_protocol_version)
    print('Local context:  ', self.context.is_local and 'Yes' or 'No')