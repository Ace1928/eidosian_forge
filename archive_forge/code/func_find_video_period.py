from collections import defaultdict
import numpy as np
from moviepy.decorators import use_clip_fps_by_default
@use_clip_fps_by_default
def find_video_period(clip, fps=None, tmin=0.3):
    """ Finds the period of a video based on frames correlation """
    frame = lambda t: clip.get_frame(t).flatten()
    tt = np.arange(tmin, clip.duration, 1.0 / fps)[1:]
    ref = frame(0)
    corrs = [np.corrcoef(ref, frame(t))[0, 1] for t in tt]
    return tt[np.argmax(corrs)]