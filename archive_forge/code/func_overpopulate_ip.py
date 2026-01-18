import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def overpopulate_ip():
    origin = Vector2(7.22, 2004.0)
    target = Vector2(12.3, 2021.0)
    origin.move_towards_ip(target, 3, 2)