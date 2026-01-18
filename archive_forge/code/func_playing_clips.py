import numpy as np
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.VideoClip import ColorClip, VideoClip
def playing_clips(self, t=0):
    """ Returns a list of the clips in the composite clips that are
            actually playing at the given time `t`. """
    return [c for c in self.clips if c.is_playing(t)]