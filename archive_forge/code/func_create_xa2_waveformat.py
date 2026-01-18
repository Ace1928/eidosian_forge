from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def create_xa2_waveformat(audio_format):
    wfx = lib.WAVEFORMATEX()
    wfx.wFormatTag = lib.WAVE_FORMAT_PCM
    wfx.nChannels = audio_format.channels
    wfx.nSamplesPerSec = audio_format.sample_rate
    wfx.wBitsPerSample = audio_format.sample_size
    wfx.nBlockAlign = wfx.wBitsPerSample * wfx.nChannels // 8
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign
    return wfx