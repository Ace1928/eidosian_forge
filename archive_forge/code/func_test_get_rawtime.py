import os
import platform
import unittest
import pygame
import time
def test_get_rawtime(self):
    iterations = 10
    delay = 0.1
    delay_miliseconds = delay * 10 ** 3
    framerate_limit = 5
    delta = 50
    c = Clock()
    self.assertEqual(c.get_rawtime(), 0)
    for f in range(iterations):
        time.sleep(delay)
        c.tick(framerate_limit)
        c1 = c.get_rawtime()
        self.assertAlmostEqual(delay_miliseconds, c1, delta=delta)
    for f in range(iterations):
        time.sleep(delay)
        c.tick()
        c1 = c.get_rawtime()
        c2 = c.get_time()
        self.assertAlmostEqual(c1, c2, delta=delta)