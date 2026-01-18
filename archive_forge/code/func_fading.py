import numpy as np
from moviepy.decorators import audio_video_fx, requires_duration
def fading(gf, t):
    gft = gf(t)
    if np.isscalar(t):
        factor = min(1.0 * (clip.duration - t) / duration, 1)
        factor = np.array([factor, factor])
    else:
        factor = np.minimum(1.0 * (clip.duration - t) / duration, 1)
        factor = np.vstack([factor, factor]).T
    return factor * gft