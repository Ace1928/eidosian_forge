from moviepy.decorators import add_mask_if_none, requires_duration
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from .CompositeVideoClip import CompositeVideoClip
@requires_duration
@add_mask_if_none
def crossfadein(clip, duration):
    """ Makes the clip appear progressively, over ``duration`` seconds.
    Only works when the clip is included in a CompositeVideoClip.
    """
    clip.mask.duration = clip.duration
    newclip = clip.copy()
    newclip.mask = clip.mask.fx(fadein, duration)
    return newclip