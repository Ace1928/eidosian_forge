import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_array_to_surface_broadcasting(self):
    targets = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
    w, h = self.surf_size
    column = pygame.Surface((1, h), 0, 32)
    for target in targets:
        source = pygame.Surface((1, h), 0, target)
        for y in range(h):
            source.set_at((0, y), pygame.Color(y + 1, y + h + 1, y + 2 * h + 1))
        pygame.pixelcopy.surface_to_array(column.get_view('2'), source)
        pygame.pixelcopy.array_to_surface(target, column.get_view('2'))
        for x in range(w):
            for y in range(h):
                self.assertEqual(target.get_at_mapped((x, y)), column.get_at_mapped((0, y)))
    row = pygame.Surface((w, 1), 0, 32)
    for target in targets:
        source = pygame.Surface((w, 1), 0, target)
        for x in range(w):
            source.set_at((x, 0), pygame.Color(x + 1, x + w + 1, x + 2 * w + 1))
        pygame.pixelcopy.surface_to_array(row.get_view('2'), source)
        pygame.pixelcopy.array_to_surface(target, row.get_view('2'))
        for x in range(w):
            for y in range(h):
                self.assertEqual(target.get_at_mapped((x, y)), row.get_at_mapped((x, 0)))
    pixel = pygame.Surface((1, 1), 0, 32)
    for target in targets:
        source = pygame.Surface((1, 1), 0, target)
        source.set_at((0, 0), pygame.Color(13, 47, 101))
        pygame.pixelcopy.surface_to_array(pixel.get_view('2'), source)
        pygame.pixelcopy.array_to_surface(target, pixel.get_view('2'))
        p = pixel.get_at_mapped((0, 0))
        for x in range(w):
            for y in range(h):
                self.assertEqual(target.get_at_mapped((x, y)), p)