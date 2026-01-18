import re
import threading
import time
from ._utils import logger
def get_output_video_line(lines):
    """Get the line that defines the video stream that ffmpeg outputs,
    and which we read.
    """
    in_output = False
    for line in lines:
        sline = line.lstrip()
        if sline.startswith(b'Output '):
            in_output = True
        elif in_output:
            if sline.startswith(b'Stream ') and b' Video:' in sline:
                return line