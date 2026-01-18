import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def is_in_subclip(t1, t2):
    try:
        return t_start <= t1 < t_end or t_start < t2 <= t_end
    except:
        return False