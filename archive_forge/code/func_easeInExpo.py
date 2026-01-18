from __future__ import division
import math
def easeInExpo(n):
    """An exponential tween function that begins slow and then accelerates.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    if n == 0:
        return 0
    else:
        return 2 ** (10 * (n - 1))