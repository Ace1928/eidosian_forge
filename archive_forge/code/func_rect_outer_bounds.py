import os
import pygame
import sys
import tempfile
import time
def rect_outer_bounds(rect):
    """

     Returns topleft outerbound if possible and then the other pts, that are
     "exclusive" bounds of the rect

    ?------O
     |RECT|      ?|0)uterbound
     |----|
    O      O

    """
    return ([(rect.left - 1, rect.top)] if rect.left else []) + [rect.topright, rect.bottomleft, rect.bottomright]