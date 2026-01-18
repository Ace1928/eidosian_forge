from __future__ import division
import math
def easeInOutSine(n):
    """A sinusoidal tween function that accelerates, reaches the midpoint, and then decelerates.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    return -0.5 * (math.cos(math.pi * n) - 1)