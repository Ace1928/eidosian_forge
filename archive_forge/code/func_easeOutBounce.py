from __future__ import division
import math
def easeOutBounce(n):
    """A bouncing tween function that hits the destination and then bounces to rest.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    if n < 1 / 2.75:
        return 7.5625 * n * n
    elif n < 2 / 2.75:
        n -= 1.5 / 2.75
        return 7.5625 * n * n + 0.75
    elif n < 2.5 / 2.75:
        n -= 2.25 / 2.75
        return 7.5625 * n * n + 0.9375
    else:
        n -= 2.65 / 2.75
        return 7.5625 * n * n + 0.984375