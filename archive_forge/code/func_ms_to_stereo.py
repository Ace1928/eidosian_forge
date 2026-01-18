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
def ms_to_stereo(audio_segment):
    """
	Mid-Side -> Left-Right
	"""
    channel = audio_segment.split_to_mono()
    channel = [channel[0].overlay(channel[1]) - 3, channel[0].overlay(channel[1].invert_phase()) - 3]
    return AudioSegment.from_mono_audiosegments(channel[0], channel[1])