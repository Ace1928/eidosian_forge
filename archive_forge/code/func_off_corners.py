from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def off_corners(rect):
    """Returns a tuple with the positions off of the corners of the given rect.

    Clockwise from the top left corner.
    """
    return ((rect.left - 1, rect.top), (rect.left - 1, rect.top - 1), (rect.left, rect.top - 1), (rect.right - 1, rect.top - 1), (rect.right, rect.top - 1), (rect.right, rect.top), (rect.right, rect.bottom - 1), (rect.right, rect.bottom), (rect.right - 1, rect.bottom), (rect.left, rect.bottom), (rect.left - 1, rect.bottom), (rect.left - 1, rect.bottom - 1))