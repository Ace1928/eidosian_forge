import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def check_not_at(self, surf, posn, color):
    sc = surf.get_at(posn)
    fail_msg = '%s != %s at %s, bitsize: %i, flags: %i, masks: %s' % (sc, color, posn, surf.get_bitsize(), surf.get_flags(), surf.get_masks())
    self.assertNotEqual(sc, color, fail_msg)