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
def get_fps():
    match = re.search('( [0-9]*.| )[0-9]* fps', line)
    fps = float(line[match.start():match.end()].split(' ')[1])
    return fps