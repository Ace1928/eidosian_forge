import os
import subprocess as sp
import numpy as np
from proglog import proglog
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting
def ffmpeg_write_image(filename, image, logfile=False):
    """ Writes an image (HxWx3 or HxWx4 numpy array) to a file, using
        ffmpeg. """
    if image.dtype != 'uint8':
        image = image.astype('uint8')
    cmd = [get_setting('FFMPEG_BINARY'), '-y', '-s', '%dx%d' % image.shape[:2][::-1], '-f', 'rawvideo', '-pix_fmt', 'rgba' if image.shape[2] == 4 else 'rgb24', '-i', '-', filename]
    if logfile:
        log_file = open(filename + '.log', 'w+')
    else:
        log_file = sp.PIPE
    popen_params = {'stdout': DEVNULL, 'stderr': log_file, 'stdin': sp.PIPE}
    if os.name == 'nt':
        popen_params['creationflags'] = 134217728
    proc = sp.Popen(cmd, **popen_params)
    out, err = proc.communicate(image.tostring())
    if proc.returncode:
        err = '\n'.join(['[MoviePy] Running : %s\n' % cmd, 'WARNING: this command returned an error:', err.decode('utf8')])
        raise IOError(err)
    del proc