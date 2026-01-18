from __future__ import division
import math
def easeOutQuint(n):
    """Starts fast and decelerates to stop. (Quintic function.)

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    n -= 1
    return n ** 5 + 1