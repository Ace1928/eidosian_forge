import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'requires a non-"dummy" SDL_VIDEODRIVER')
def test_image_convert_bug_131(self):
    pygame.display.init()
    try:
        pygame.display.set_mode((640, 480))
        im = pygame.image.load(example_path(os.path.join('data', 'city.png')))
        im2 = pygame.image.load(example_path(os.path.join('data', 'brick.png')))
        self.assertEqual(im.get_palette(), ((0, 0, 0, 255), (255, 255, 255, 255)))
        self.assertEqual(im2.get_palette(), ((0, 0, 0, 255), (0, 0, 0, 255)))
        self.assertEqual(repr(im.convert(32)), '<Surface(24x24x32 SW)>')
        self.assertEqual(repr(im2.convert(32)), '<Surface(469x137x32 SW)>')
        im3 = im.convert(8)
        self.assertEqual(repr(im3), '<Surface(24x24x8 SW)>')
        self.assertEqual(im3.get_palette(), im.get_palette())
    finally:
        pygame.display.quit()