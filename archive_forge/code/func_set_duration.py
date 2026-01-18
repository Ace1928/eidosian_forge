from copy import copy
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.decorators import (apply_to_audio, apply_to_mask,
@apply_to_mask
@apply_to_audio
@convert_to_seconds(['t'])
@outplace
def set_duration(self, t, change_end=True):
    """
        Returns a copy of the clip, with the  ``duration`` attribute
        set to ``t``, which can be expressed in seconds (15.35), in (min, sec),
        in (hour, min, sec), or as a string: '01:03:05.35'.
        Also sets the duration of the mask and audio, if any, of the
        returned clip.
        If change_end is False, the start attribute of the clip will
        be modified in function of the duration and the preset end
        of the clip.
        """
    self.duration = t
    if change_end:
        self.end = None if t is None else self.start + t
    else:
        if self.duration is None:
            raise Exception('Cannot change clip start when newduration is None')
        self.start = self.end - t