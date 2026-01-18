from __future__ import division
import math
def easeInOutQuad(n):
    """Accelerates, reaches the midpoint, and then decelerates. (Quadratic function.)

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    if n < 0.5:
        return 2 * n ** 2
    else:
        n = n * 2 - 1
        return -0.5 * (n * (n - 2) - 1)