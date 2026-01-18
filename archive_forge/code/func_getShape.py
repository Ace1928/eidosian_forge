import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
def getShape(self):
    """Returns a tuple (T, M, N, C)

        Returns the video shape in number of frames, height, width, and channels per pixel.
        """
    return (self.inputframenum, self.outputheight, self.outputwidth, self.outputdepth)