from __future__ import division
import logging
import os
import re
import subprocess as sp
import warnings
import numpy as np
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting  # ffmpeg, ffmpeg.exe, etc...
from moviepy.tools import cvsecs
def get_tbr():
    match = re.search('( [0-9]*.| )[0-9]* tbr', line)
    s_tbr = line[match.start():match.end()].split(' ')[1]
    if 'k' in s_tbr:
        tbr = float(s_tbr.replace('k', '')) * 1000
    else:
        tbr = float(s_tbr)
    return tbr