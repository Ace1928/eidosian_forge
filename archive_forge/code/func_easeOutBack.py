from __future__ import division
import math
def easeOutBack(n, s=1.70158):
    """A tween function that overshoots the destination a little and then backs into the destination.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    n -= 1
    return n * n * ((s + 1) * n + s) + 1