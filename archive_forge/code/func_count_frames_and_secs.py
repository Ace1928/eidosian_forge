import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from ._parsing import LogCatcher, cvsecs, parse_ffmpeg_header
from ._utils import _popen_kwargs, get_ffmpeg_exe, logger
def count_frames_and_secs(path):
    """
    Get the number of frames and number of seconds for the given video
    file. Note that this operation can be quite slow for large files.

    Disclaimer: I've seen this produce different results from actually reading
    the frames with older versions of ffmpeg (2.x). Therefore I cannot say
    with 100% certainty that the returned values are always exact.
    """
    if isinstance(path, pathlib.PurePath):
        path = str(path)
    if not isinstance(path, str):
        raise TypeError('Video path must be a string or pathlib.Path.')
    cmd = [get_ffmpeg_exe(), '-i', path, '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-']
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, **_popen_kwargs())
    except subprocess.CalledProcessError as err:
        out = err.output.decode(errors='ignore')
        raise RuntimeError('FFMPEG call failed with {}:\n{}'.format(err.returncode, out))
    nframes = nsecs = None
    for line in reversed(out.splitlines()):
        if line.startswith(b'frame='):
            line = line.decode(errors='ignore')
            i = line.find('frame=')
            if i >= 0:
                s = line[i:].split('=', 1)[-1].lstrip().split(' ', 1)[0].strip()
                nframes = int(s)
            i = line.find('time=')
            if i >= 0:
                s = line[i:].split('=', 1)[-1].lstrip().split(' ', 1)[0].strip()
                nsecs = cvsecs(*s.split(':'))
            return (nframes, nsecs)
    raise RuntimeError('Could not get number of frames')