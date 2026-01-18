import os
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
from moviepy.Clip import Clip
from moviepy.decorators import requires_duration
from moviepy.tools import deprecated_version_of, extensions_dict
def concatenate_audioclips(clips):
    """
    The clip with the highest FPS will be the FPS of the result clip.
    """
    durations = [c.duration for c in clips]
    tt = np.cumsum([0] + durations)
    newclips = [c.set_start(t) for c, t in zip(clips, tt)]
    result = CompositeAudioClip(newclips).set_duration(tt[-1])
    fpss = [c.fps for c in clips if getattr(c, 'fps', None)]
    result.fps = max(fpss) if fpss else None
    return result