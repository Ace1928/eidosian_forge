from moviepy.decorators import add_mask_if_none, requires_duration
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from .CompositeVideoClip import CompositeVideoClip
def slide_in(clip, duration, side):
    """ Makes the clip arrive from one side of the screen.

    Only works when the clip is included in a CompositeVideoClip,
    and if the clip has the same size as the whole composition.

    Parameters
    ===========

    clip
      A video clip.

    duration
      Time taken for the clip to be fully visible

    side
      Side of the screen where the clip comes from. One of
      'top' | 'bottom' | 'left' | 'right'

    Examples
    =========

    >>> from moviepy.editor import *
    >>> clips = [... make a list of clips]
    >>> slided_clips = [CompositeVideoClip([
                            clip.fx(transfx.slide_in, duration=1, side='left')])
                        for clip in clips]
    >>> final_clip = concatenate( slided_clips, padding=-1)

    """
    w, h = clip.size
    pos_dict = {'left': lambda t: (min(0, w * (t / duration - 1)), 'center'), 'right': lambda t: (max(0, w * (1 - t / duration)), 'center'), 'top': lambda t: ('center', min(0, h * (t / duration - 1))), 'bottom': lambda t: ('center', max(0, h * (1 - t / duration)))}
    return clip.set_position(pos_dict[side])