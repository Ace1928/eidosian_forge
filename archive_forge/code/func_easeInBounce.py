from __future__ import division
import math
def easeInBounce(n):
    """A bouncing tween function that begins bouncing and then jumps to the destination.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    return 1 - easeOutBounce(1 - n)