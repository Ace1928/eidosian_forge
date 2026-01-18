import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_distance_to_exceptions(self):
    v2 = Vector2(10, 10)
    v3 = Vector3(1, 1, 1)
    self.assertRaises(ValueError, v2.distance_to, v3)
    self.assertRaises(ValueError, v3.distance_to, v2)
    self.assertRaises(ValueError, v2.distance_to, (1, 1, 1))
    self.assertRaises(ValueError, v2.distance_to, (1, 1, 0))
    self.assertRaises(ValueError, v2.distance_to, (1,))
    self.assertRaises(ValueError, v2.distance_to, [1, 1, 1])
    self.assertRaises(ValueError, v2.distance_to, [1, 1, 0])
    self.assertRaises(ValueError, v2.distance_to, [1])
    self.assertRaises(ValueError, v2.distance_to, (1, 1, 1))
    self.assertRaises(ValueError, v3.distance_to, (1, 1))
    self.assertRaises(ValueError, v3.distance_to, (1,))
    self.assertRaises(ValueError, v3.distance_to, [1, 1])
    self.assertRaises(ValueError, v3.distance_to, [1])
    self.assertRaises(TypeError, v2.distance_to, (1, 'hello'))
    self.assertRaises(TypeError, v2.distance_to, ([], []))
    self.assertRaises(TypeError, v2.distance_to, (1, ('hello',)))
    self.assertRaises(TypeError, v2.distance_to)
    self.assertRaises(TypeError, v2.distance_to, (1, 1), (1, 2))
    self.assertRaises(TypeError, v2.distance_to, (1, 1), (1, 2), 1)