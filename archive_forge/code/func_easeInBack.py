from __future__ import division
import math
def easeInBack(n, s=1.70158):
    """A tween function that backs up first at the start and then goes to the destination.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    return n * n * ((s + 1) * n - s)