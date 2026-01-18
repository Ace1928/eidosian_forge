import os
import subprocess as sp
import warnings
import numpy as np
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
def close_proc(self):
    if hasattr(self, 'proc') and self.proc is not None:
        self.proc.terminate()
        for std in [self.proc.stdout, self.proc.stderr]:
            std.close()
        self.proc.wait()
        self.proc = None