import os
import subprocess as sp
import sys
from moviepy.config import get_setting
from moviepy.tools import subprocess_call
def ffmpeg_movie_from_frames(filename, folder, fps, digits=6, bitrate='v'):
    """
    Writes a movie out of the frames (picture files) in a folder.
    Almost deprecated.
    """
    s = '%' + '%02d' % digits + 'd.png'
    cmd = [get_setting('FFMPEG_BINARY'), '-y', '-f', 'image2', '-r', '%d' % fps, '-i', os.path.join(folder, folder) + '/' + s, '-b', '%dk' % bitrate, '-r', '%d' % fps, filename]
    subprocess_call(cmd)