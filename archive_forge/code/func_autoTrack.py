import numpy as np
from moviepy.decorators import convert_to_seconds, use_clip_fps_by_default
from ..io.preview import imdisplay
from .interpolators import Trajectory
def autoTrack(clip, pattern, tt=None, fps=None, radius=20, xy0=None):
    """
    Tracks a given pattern (small image array) in a video clip.
    Returns [(x1,y1),(x2,y2)...] where xi,yi are
    the coordinates of the pattern in the clip on frame i.
    To select the frames you can either specify a list of times with ``tt``
    or select a frame rate with ``fps``.
    This algorithm assumes that the pattern's aspect does not vary much
    and that the distance between two occurences of the pattern in
    two consecutive frames is smaller than ``radius`` (if you set ``radius``
    to -1 the pattern will be searched in the whole screen at each frame).
    You can also provide the original position of the pattern with xy0.
    """
    if not autotracking_possible:
        raise IOError('Sorry, autotrack requires OpenCV for the moment. Install OpenCV (aka cv2) to use it.')
    if not xy0:
        xy0 = findAround(clip.get_frame(tt[0]), pattern)
    if tt is None:
        tt = np.arange(0, clip.duration, 1.0 / fps)
    xys = [xy0]
    for t in tt[1:]:
        xys.append(findAround(clip.get_frame(t), pattern, xy=xys[-1], r=radius))
    xx, yy = zip(*xys)
    return Trajectory(tt, xx, yy)