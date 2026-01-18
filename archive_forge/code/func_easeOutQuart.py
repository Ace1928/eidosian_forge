from __future__ import division
import math
def easeOutQuart(n):
    """Starts fast and decelerates to stop. (Quartic function.)

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    n -= 1
    return -(n ** 4 - 1)