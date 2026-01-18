from __future__ import division
import math
def easeInOutExpo(n):
    """An exponential tween function that accelerates, reaches the midpoint, and then decelerates.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        n *= 2
        if n < 1:
            return 0.5 * 2 ** (10 * (n - 1))
        else:
            n -= 1
            return 0.5 * (-1 * 2 ** (-10 * n) + 2)