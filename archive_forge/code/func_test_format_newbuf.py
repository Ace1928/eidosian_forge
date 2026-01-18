import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_format_newbuf(self):
    Exporter = self.buftools.Exporter
    surface = self.surface
    shape = surface.get_size()
    w, h = shape
    for format in ['=i', '=I', '=l', '=L', '=q', '=Q', '<i', '>i', '!i', '1i', '=1i', '@q', 'q', '4x', '8x']:
        surface.fill((255, 254, 253))
        exp = Exporter(shape, format=format)
        exp._buf[:] = [42] * exp.buflen
        array_to_surface(surface, exp)
        for x in range(w):
            for y in range(h):
                self.assertEqual(surface.get_at((x, y)), (42, 42, 42, 255))
    for format in ['f', 'd', '?', 'x', '1x', '2x', '3x', '5x', '6x', '7x', '9x']:
        exp = Exporter(shape, format=format)
        self.assertRaises(ValueError, array_to_surface, surface, exp)