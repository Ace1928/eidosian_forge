from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def get_source_voice(self, audio_format, player):
    """Get a source voice from the pool. Source voice creation can be slow to create/destroy.
        So pooling is recommended. We pool based on audio channels.
        A source voice handles all of the audio playing and state for a single source."""
    voice_key = (audio_format.channels, audio_format.sample_size)
    if not self._voice_pool[voice_key]:
        voice = self._create_new_voice(audio_format)
        self._voice_pool[voice_key].append(self._create_new_voice(audio_format))
    else:
        voice = self._voice_pool[voice_key].pop()
    assert voice.buffers_queued == 0
    voice.acquired(player.on_buffer_end, audio_format.sample_rate)
    if voice.is_emitter:
        self._emitting_voices.append(voice)
    self._in_use[voice] = player
    return voice