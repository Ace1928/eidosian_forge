import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def iter_surface_to_array_3d(self, rgba_masks):
    dst = pygame.Surface(self.surf_size, 0, 24, masks=rgba_masks)
    for surf in self.sources:
        dst.fill((0, 0, 0, 0))
        src_bitsize = surf.get_bitsize()
        view = dst.get_view('3')
        self.assertFalse(surf.get_locked())
        surface_to_array(view, surf)
        self.assertFalse(surf.get_locked())
        for posn, i in self.test_points:
            sc = surf.get_at(posn)[0:3]
            dc = dst.get_at(posn)[0:3]
            self.assertEqual(dc, sc, '%s != %s: flags: %i, bpp: %i, posn: %s' % (dc, sc, surf.get_flags(), surf.get_bitsize(), posn))
        view = None