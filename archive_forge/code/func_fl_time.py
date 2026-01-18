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
@outplace
def fl_time(self, time_func, apply_to=None, keep_duration=False):
    """Time-transformation filter.

        Applies a transformation to the clip's timeline
        (see Clip.fl_time).

        This method does nothing for ImageClips (but it may affect their
        masks or their audios). The result is still an ImageClip.
        """
    if apply_to is None:
        apply_to = ['mask', 'audio']
    for attr in apply_to:
        a = getattr(self, attr, None)
        if a is not None:
            new_a = a.fl_time(time_func)
            setattr(self, attr, new_a)