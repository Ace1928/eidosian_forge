from __future__ import division
import math
def easeOutElastic(n, amplitude=1, period=0.3):
    """An elastic tween function that overshoots the destination and then "rubber bands" into the destination.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    if amplitude < 1:
        amplitude = 1
        s = period / 4
    else:
        s = period / (2 * math.pi) * math.asin(1 / amplitude)
    return amplitude * 2 ** (-10 * n) * math.sin((n - s) * (2 * math.pi / period)) + 1