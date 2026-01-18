import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def assign_nonfloat():
    v = Vector2()
    v[0] = 'spam'