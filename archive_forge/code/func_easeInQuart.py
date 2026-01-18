from __future__ import division
import math
def easeInQuart(n):
    """Starts fast and decelerates. (Quartic function.)

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    return n ** 4