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
def skip_frames(self, n=1):
    """Reads and throws away n frames """
    w, h = self.size
    for i in range(n):
        self.proc.stdout.read(self.depth * w * h)
    self.pos += n