from __future__ import division
import math
def easeInOutBounce(n):
    """A bouncing tween function that bounces at the start and end.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    if n < 0.5:
        return easeInBounce(n * 2) * 0.5
    else:
        return easeOutBounce(n * 2 - 1) * 0.5 + 0.5