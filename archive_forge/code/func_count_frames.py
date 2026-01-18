import re
import sys
import time
import logging
import platform
import threading
import subprocess as sp
import imageio_ffmpeg
import numpy as np
from ..core import Format, image_as_uint
def count_frames(self):
    """Count the number of frames. Note that this can take a few
            seconds for large files. Also note that it counts the number
            of frames in the original video and does not take a given fps
            into account.
            """
    cf = self._ffmpeg_api.count_frames_and_secs
    return cf(self._filename)[0]