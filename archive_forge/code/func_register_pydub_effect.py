from __future__ import division
import json
import os
import re
import sys
from subprocess import Popen, PIPE
from math import log, ceil
from tempfile import TemporaryFile
from warnings import warn
from functools import wraps
def register_pydub_effect(fn, name=None):
    """
    decorator for adding pydub effects to the AudioSegment objects.
    example use:
        @register_pydub_effect
        def normalize(audio_segment):
            ...
    or you can specify a name:
        @register_pydub_effect("normalize")
        def normalize_audio_segment(audio_segment):
            ...
    """
    if isinstance(fn, basestring):
        name = fn
        return lambda fn: register_pydub_effect(fn, name)
    if name is None:
        name = fn.__name__
    from .audio_segment import AudioSegment
    setattr(AudioSegment, name, fn)
    return fn