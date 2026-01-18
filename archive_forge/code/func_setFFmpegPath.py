from .utils import check_output, where
import os
import warnings
import numpy as np
def setFFmpegPath(path):
    """ Sets up the path to the directory containing both ffmpeg and ffprobe

        Use this function for to specify specific system installs of FFmpeg. All
        calls to ffmpeg and ffprobe will use this path as a prefix.

        Parameters
        ----------
        path : string
            Path to directory containing ffmpeg and ffprobe

        Returns
        -------
        none

    """
    global _FFMPEG_PATH
    global _HAS_FFMPEG
    _FFMPEG_PATH = path
    if os.path.isfile(os.path.join(_FFMPEG_PATH, _FFMPEG_APPLICATION)) and os.path.isfile(os.path.join(_FFMPEG_PATH, _FFPROBE_APPLICATION)):
        _HAS_FFMPEG = 1
    else:
        warnings.warn('ffmpeg/ffprobe not found in path: ' + str(path), UserWarning)
        _HAS_FFMPEG = 0
        global _FFMPEG_MAJOR_VERSION
        global _FFMPEG_MINOR_VERSION
        global _FFMPEG_PATCH_VERSION
        _FFMPEG_MAJOR_VERSION = '0'
        _FFMPEG_MINOR_VERSION = '0'
        _FFMPEG_PATCH_VERSION = '0'
        return
    scan_ffmpeg()