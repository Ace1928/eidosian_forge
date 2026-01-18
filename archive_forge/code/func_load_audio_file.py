from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
def load_audio_file(*args, **kwargs):
    """
    Deprecated as of version 0.16. Please use madmom.io.audio.load_audio_file
    instead. Will be removed in version 0.18.

    """
    warnings.warn('Deprecated as of version 0.16. Please use madmom.io.audio.load_audio_file instead. Will be removed in version 0.18.')
    from ..io.audio import load_audio_file
    return load_audio_file(*args, **kwargs)