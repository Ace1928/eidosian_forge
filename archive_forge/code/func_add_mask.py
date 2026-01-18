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
def add_mask(self):
    """Add a mask VideoClip to the VideoClip.

        Returns a copy of the clip with a completely opaque mask
        (made of ones). This makes computations slower compared to
        having a None mask but can be useful in many cases. Choose

        Set ``constant_size`` to  `False` for clips with moving
        image size.
        """
    if self.has_constant_size:
        mask = ColorClip(self.size, 1.0, ismask=True)
        return self.set_mask(mask.set_duration(self.duration))
    else:
        make_frame = lambda t: np.ones(self.get_frame(t).shape[:2], dtype=float)
        mask = VideoClip(ismask=True, make_frame=make_frame)
        return self.set_mask(mask.set_duration(self.duration))