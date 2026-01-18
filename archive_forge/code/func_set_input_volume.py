import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def set_input_volume(self, stream: 'PulseAudioStream', volume: float) -> 'PulseAudioOperation':
    """
        Set the volume for a stream.
        """
    cvolume = self._get_cvolume_from_linear(stream, volume)
    clump = PulseAudioContextSuccessCallbackLump(self)
    return PulseAudioOperation(clump, pa.pa_context_set_sink_input_volume(self._pa_context, stream.index, cvolume, clump.pa_callback, None))