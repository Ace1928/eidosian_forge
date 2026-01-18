import os
import subprocess as sp
import proglog
from moviepy.compat import DEVNULL
from moviepy.config import get_setting
from moviepy.decorators import requires_duration
@requires_duration
def ffmpeg_audiowrite(clip, filename, fps, nbytes, buffersize, codec='libvorbis', bitrate=None, write_logfile=False, verbose=True, ffmpeg_params=None, logger='bar'):
    """
    A function that wraps the FFMPEG_AudioWriter to write an AudioClip
    to a file.

    NOTE: verbose is deprecated.
    """
    if write_logfile:
        logfile = open(filename + '.log', 'w+')
    else:
        logfile = None
    logger = proglog.default_bar_logger(logger)
    logger(message='MoviePy - Writing audio in %s' % filename)
    writer = FFMPEG_AudioWriter(filename, fps, nbytes, clip.nchannels, codec=codec, bitrate=bitrate, logfile=logfile, ffmpeg_params=ffmpeg_params)
    for chunk in clip.iter_chunks(chunksize=buffersize, quantize=True, nbytes=nbytes, fps=fps, logger=logger):
        writer.write_frames(chunk)
    writer.close()
    if write_logfile:
        logfile.close()
    logger(message='MoviePy - Done.')