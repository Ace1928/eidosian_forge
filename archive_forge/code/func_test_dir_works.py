import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_dir_works(self):
    attributes = {'lerp', 'normalize', 'normalize_ip', 'reflect', 'slerp', 'x', 'y'}
    self.assertTrue(attributes.issubset(set(dir(self.v1))))