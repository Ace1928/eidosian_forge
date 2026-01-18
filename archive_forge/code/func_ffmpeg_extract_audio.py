import os
import subprocess as sp
import sys
from moviepy.config import get_setting
from moviepy.tools import subprocess_call
def ffmpeg_extract_audio(inputfile, output, bitrate=3000, fps=44100):
    """ extract the sound from a video file and save it in ``output`` """
    cmd = [get_setting('FFMPEG_BINARY'), '-y', '-i', inputfile, '-ab', '%dk' % bitrate, '-ar', '%d' % fps, output]
    subprocess_call(cmd)