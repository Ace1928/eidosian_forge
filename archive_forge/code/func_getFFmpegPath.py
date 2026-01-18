from .utils import check_output, where
import os
import warnings
import numpy as np
def getFFmpegPath():
    """ Returns the path to the directory containing both ffmpeg and ffprobe 
    """
    return _FFMPEG_PATH