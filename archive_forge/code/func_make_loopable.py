from moviepy.decorators import add_mask_if_none, requires_duration
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from .CompositeVideoClip import CompositeVideoClip
@requires_duration
def make_loopable(clip, cross_duration):
    """ Makes the clip fade in progressively at its own end, this way
    it can be looped indefinitely. ``cross`` is the duration in seconds
    of the fade-in.  """
    d = clip.duration
    clip2 = clip.fx(crossfadein, cross_duration).set_start(d - cross_duration)
    return CompositeVideoClip([clip, clip2]).subclip(cross_duration, d)