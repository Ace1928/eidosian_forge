import numpy as np
from moviepy.decorators import convert_to_seconds, use_clip_fps_by_default
from ..io.preview import imdisplay
from .interpolators import Trajectory
def findAround(pic, pat, xy=None, r=None):
    """
    find image pattern ``pat`` in ``pic[x +/- r, y +/- r]``.
    if xy is none, consider the whole picture.
    """
    if xy and r:
        h, w = pat.shape[:2]
        x, y = xy
        pic = pic[y - r:y + h + r, x - r:x + w + r]
    matches = cv2.matchTemplate(pat, pic, cv2.TM_CCOEFF_NORMED)
    yf, xf = np.unravel_index(matches.argmax(), matches.shape)
    return (x - r + xf, y - r + yf) if xy and r else (xf, yf)