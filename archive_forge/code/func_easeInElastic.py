from __future__ import division
import math
def easeInElastic(n, amplitude=1, period=0.3):
    """An elastic tween function that begins with an increasing wobble and then snaps into the destination.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    return 1 - easeOutElastic(1 - n, amplitude=amplitude, period=period)