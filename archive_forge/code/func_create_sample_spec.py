import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def create_sample_spec(self, audio_format: 'AudioFormat') -> pa.pa_sample_spec:
    """
        Create a PulseAudio sample spec from pyglet audio format.
        """
    _FORMATS = {('little', 8): pa.PA_SAMPLE_U8, ('big', 8): pa.PA_SAMPLE_U8, ('little', 16): pa.PA_SAMPLE_S16LE, ('big', 16): pa.PA_SAMPLE_S16BE, ('little', 24): pa.PA_SAMPLE_S24LE, ('big', 24): pa.PA_SAMPLE_S24BE}
    fmt = (sys.byteorder, audio_format.sample_size)
    if fmt not in _FORMATS:
        raise MediaException(f'Unsupported sample size/format: {fmt}')
    sample_spec = pa.pa_sample_spec()
    sample_spec.format = _FORMATS[fmt]
    sample_spec.rate = audio_format.sample_rate
    sample_spec.channels = audio_format.channels
    return sample_spec