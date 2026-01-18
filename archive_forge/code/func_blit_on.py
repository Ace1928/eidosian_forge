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
def blit_on(self, picture, t):
    """
        Returns the result of the blit of the clip's frame at time `t`
        on the given `picture`, the position of the clip being given
        by the clip's ``pos`` attribute. Meant for compositing.
        """
    hf, wf = framesize = picture.shape[:2]
    if self.ismask and picture.max():
        return np.minimum(1, picture + self.blit_on(np.zeros(framesize), t))
    ct = t - self.start
    img = self.get_frame(ct)
    mask = self.mask.get_frame(ct) if self.mask else None
    if mask is not None and (img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]):
        img = self.fill_array(img, mask.shape)
    hi, wi = img.shape[:2]
    pos = self.pos(ct)
    if isinstance(pos, str):
        pos = {'center': ['center', 'center'], 'left': ['left', 'center'], 'right': ['right', 'center'], 'top': ['center', 'top'], 'bottom': ['center', 'bottom']}[pos]
    else:
        pos = list(pos)
    if self.relative_pos:
        for i, dim in enumerate([wf, hf]):
            if not isinstance(pos[i], str):
                pos[i] = dim * pos[i]
    if isinstance(pos[0], str):
        D = {'left': 0, 'center': (wf - wi) / 2, 'right': wf - wi}
        pos[0] = D[pos[0]]
    if isinstance(pos[1], str):
        D = {'top': 0, 'center': (hf - hi) / 2, 'bottom': hf - hi}
        pos[1] = D[pos[1]]
    pos = map(int, pos)
    return blit(img, picture, pos, mask=mask, ismask=self.ismask)