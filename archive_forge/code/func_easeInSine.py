from __future__ import division
import math
def easeInSine(n):
    """A sinusoidal tween function that begins slow and then accelerates.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    return -1 * math.cos(n * math.pi / 2) + 1