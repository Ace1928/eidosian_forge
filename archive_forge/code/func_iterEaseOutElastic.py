from __future__ import division
import math
def iterEaseOutElastic(startX, startY, endX, endY, intervalSize, amplitude=1, period=0.3):
    """Returns an iterator of a easeOutElastic tween between the start and end points, incrementing the
    interpolation factor by intervalSize each time. Guaranteed to return the point for 0.0 first
    and 1.0 last no matter the intervalSize."""
    return iter(_iterTween(startX, startY, endX, endY, intervalSize, easeOutElastic, amplitude, period))