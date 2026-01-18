import numpy as np
from moviepy.decorators import apply_to_mask
from moviepy.video.VideoClip import ImageClip
def make_bg(w, h):
    new_w, new_h = (w + left + right, h + top + bottom)
    if clip.ismask:
        shape = (new_h, new_w)
        bg = np.tile(opacity, (new_h, new_w)).astype(float).reshape(shape)
    else:
        shape = (new_h, new_w, 3)
        bg = np.tile(color, (new_h, new_w)).reshape(shape)
    return bg