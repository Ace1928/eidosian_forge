import os
import subprocess as sp
import tempfile
import warnings
import numpy as np
import proglog
from imageio import imread, imsave
from ..Clip import Clip
from ..compat import DEVNULL, string_types
from ..config import get_setting
from ..decorators import (add_mask_if_none, apply_to_mask,
from ..tools import (deprecated_version_of, extensions_dict, find_extension,
from .io.ffmpeg_writer import ffmpeg_write_video
from .io.gif_writers import (write_gif, write_gif_with_image_io,
from .tools.drawing import blit
@convert_to_seconds(['t'])
@convert_masks_to_RGB
def save_frame(self, filename, t=0, withmask=True):
    """ Save a clip's frame to an image file.

        Saves the frame of clip corresponding to time ``t`` in
        'filename'. ``t`` can be expressed in seconds (15.35), in
        (min, sec), in (hour, min, sec), or as a string: '01:03:05.35'.

        If ``withmask`` is ``True`` the mask is saved in
        the alpha layer of the picture (only works with PNGs).

        """
    im = self.get_frame(t)
    if withmask and self.mask is not None:
        mask = 255 * self.mask.get_frame(t)
        im = np.dstack([im, mask]).astype('uint8')
    else:
        im = im.astype('uint8')
    imsave(filename, im)