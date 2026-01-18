import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def file_to_subtitles(filename):
    """ Converts a srt file into subtitles.

    The returned list is of the form ``[((ta,tb),'some text'),...]``
    and can be fed to SubtitlesClip.

    Only works for '.srt' format for the moment.
    """
    times_texts = []
    current_times = None
    current_text = ''
    with open(filename, 'r') as f:
        for line in f:
            times = re.findall('([0-9]*:[0-9]*:[0-9]*,[0-9]*)', line)
            if times:
                current_times = [cvsecs(t) for t in times]
            elif line.strip() == '':
                times_texts.append((current_times, current_text.strip('\n')))
                current_times, current_text = (None, '')
            elif current_times:
                current_text += line
    return times_texts