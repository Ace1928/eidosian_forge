from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def enable_3d(self):
    """Initializes the prerequisites for 3D positional audio and initializes with default DSP settings."""
    channel_mask = DWORD()
    self._master_voice.GetChannelMask(byref(channel_mask))
    self._x3d_handle = lib.X3DAUDIO_HANDLE()
    lib.X3DAudioInitialize(channel_mask.value, lib.X3DAUDIO_SPEED_OF_SOUND, self._x3d_handle)
    matrix = (FLOAT * self._mvoice_details.InputChannels)()
    self._dsp_settings = lib.X3DAUDIO_DSP_SETTINGS()
    self._dsp_settings.SrcChannelCount = 1
    self._dsp_settings.DstChannelCount = self._mvoice_details.InputChannels
    self._dsp_settings.pMatrixCoefficients = matrix
    pyglet.clock.schedule_interval_soft(self._calculate_3d_sources, 1 / 15.0)