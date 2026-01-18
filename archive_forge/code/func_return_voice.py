from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def return_voice(self, voice, remaining_data):
    """Reset a voice and eventually return it to the pool. The voice must be stopped.
        `remaining_data` should contain the data this voice's remaining
        buffers point to.
        It will be `.clear()`ed shortly after as soon as the flush initiated
        by the driver completes in order to not have theoretical dangling
        pointers.
        """
    if voice.is_emitter:
        self._emitting_voices.remove(voice)
    self._in_use.pop(voice)
    assert _debug(f'XA2AudioDriver: Resetting {voice}...')
    voice_key = (voice.channel_count, voice.sample_size)
    resetter = _VoiceResetter(self, voice, voice_key, remaining_data)
    self._resetting_voices[voice] = resetter
    resetter.run()