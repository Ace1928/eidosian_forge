from __future__ import division
import math
def easeInOutElastic(n, amplitude=1, period=0.5):
    """An elastic tween function wobbles towards the midpoint.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    n *= 2
    if n < 1:
        return easeInElastic(n, amplitude=amplitude, period=period) / 2
    else:
        return easeOutElastic(n - 1, amplitude=amplitude, period=period) / 2 + 0.5