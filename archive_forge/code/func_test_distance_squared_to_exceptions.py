import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_distance_squared_to_exceptions(self):
    v2 = Vector2(10, 10)
    v3 = Vector3(1, 1, 1)
    dist_t = v2.distance_squared_to
    dist_t3 = v3.distance_squared_to
    self.assertRaises(ValueError, dist_t, v3)
    self.assertRaises(ValueError, dist_t3, v2)
    self.assertRaises(ValueError, dist_t, (1, 1, 1))
    self.assertRaises(ValueError, dist_t, (1, 1, 0))
    self.assertRaises(ValueError, dist_t, (1,))
    self.assertRaises(ValueError, dist_t, [1, 1, 1])
    self.assertRaises(ValueError, dist_t, [1, 1, 0])
    self.assertRaises(ValueError, dist_t, [1])
    self.assertRaises(ValueError, dist_t, (1, 1, 1))
    self.assertRaises(ValueError, dist_t3, (1, 1))
    self.assertRaises(ValueError, dist_t3, (1,))
    self.assertRaises(ValueError, dist_t3, [1, 1])
    self.assertRaises(ValueError, dist_t3, [1])
    self.assertRaises(TypeError, dist_t, (1, 'hello'))
    self.assertRaises(TypeError, dist_t, ([], []))
    self.assertRaises(TypeError, dist_t, (1, ('hello',)))
    self.assertRaises(TypeError, dist_t)
    self.assertRaises(TypeError, dist_t, (1, 1), (1, 2))
    self.assertRaises(TypeError, dist_t, (1, 1), (1, 2), 1)