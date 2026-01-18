import os
import pygame
import sys
import tempfile
import time
def rect_perimeter_pts(rect):
    """

    Returns pts ((L, T) tuples) encompassing the perimeter of a rect.

    The order is clockwise:

          topleft to topright
         topright to bottomright
      bottomright to bottomleft
       bottomleft to topleft

    Duplicate pts are not returned

    """
    clock_wise_from_top_left = ([(l, rect.top) for l in range(rect.left, rect.right)], [(rect.right - 1, t) for t in range(rect.top + 1, rect.bottom)], [(l, rect.bottom - 1) for l in range(rect.right - 2, rect.left - 1, -1)], [(rect.left, t) for t in range(rect.bottom - 2, rect.top, -1)])
    for line in clock_wise_from_top_left:
        yield from line