import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_move_towards_errors(self):
    origin = Vector3(7.22, 2004.0, 4.1)
    target = Vector3(12.3, 2021.0, -421.5)
    self.assertRaises(TypeError, origin.move_towards, target, 3, 2)
    self.assertRaises(TypeError, origin.move_towards_ip, target, 3, 2)
    self.assertRaises(TypeError, origin.move_towards, target, 'a')
    self.assertRaises(TypeError, origin.move_towards_ip, target, 'b')
    self.assertRaises(TypeError, origin.move_towards, 'c', 3)
    self.assertRaises(TypeError, origin.move_towards_ip, 'd', 3)