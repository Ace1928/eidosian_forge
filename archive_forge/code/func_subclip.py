from copy import copy
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.decorators import (apply_to_audio, apply_to_mask,
@convert_to_seconds(['t_start', 't_end'])
@apply_to_mask
@apply_to_audio
def subclip(self, t_start=0, t_end=None):
    """
        Returns a clip playing the content of the current clip
        between times ``t_start`` and ``t_end``, which can be expressed
        in seconds (15.35), in (min, sec), in (hour, min, sec), or as a
        string: '01:03:05.35'.
        If ``t_end`` is not provided, it is assumed to be the duration
        of the clip (potentially infinite).
        If ``t_end`` is a negative value, it is reset to
        ``clip.duration + t_end. ``. For instance: ::

            >>> # cut the last two seconds of the clip:
            >>> newclip = clip.subclip(0,-2)

        If ``t_end`` is provided or if the clip has a duration attribute,
        the duration of the returned clip is set automatically.

        The ``mask`` and ``audio`` of the resulting subclip will be
        subclips of ``mask`` and ``audio`` the original clip, if
        they exist.
        """
    if t_start < 0:
        t_start = self.duration + t_start
    if self.duration is not None and t_start > self.duration:
        raise ValueError('t_start (%.02f) ' % t_start + "should be smaller than the clip's " + 'duration (%.02f).' % self.duration)
    newclip = self.fl_time(lambda t: t + t_start, apply_to=[])
    if t_end is None and self.duration is not None:
        t_end = self.duration
    elif t_end is not None and t_end < 0:
        if self.duration is None:
            print('Error: subclip with negative times (here %s)' % str((t_start, t_end)) + ' can only be extracted from clips with a ``duration``')
        else:
            t_end = self.duration + t_end
    if t_end is not None:
        newclip.duration = t_end - t_start
        newclip.end = newclip.start + newclip.duration
    return newclip