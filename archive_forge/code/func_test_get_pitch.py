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
def test_get_pitch(self):
    sizes = ((2, 2), (7, 33), (33, 7), (2, 734), (734, 2), (734, 734))
    depths = [8, 24, 32]
    for width, height in sizes:
        for depth in depths:
            surf = pygame.Surface((width, height), depth=depth)
            buff = surf.get_buffer()
            pitch = buff.length / surf.get_height()
            test_pitch = surf.get_pitch()
            self.assertEqual(pitch, test_pitch)
            rect1 = surf.get_rect()
            subsurf1 = surf.subsurface(rect1)
            sub_buff1 = subsurf1.get_buffer()
            sub_pitch1 = sub_buff1.length / subsurf1.get_height()
            test_sub_pitch1 = subsurf1.get_pitch()
            self.assertEqual(sub_pitch1, test_sub_pitch1)
            rect2 = rect1.inflate(-width / 2, -height / 2)
            subsurf2 = surf.subsurface(rect2)
            sub_buff2 = subsurf2.get_buffer()
            sub_pitch2 = sub_buff2.length / float(subsurf2.get_height())
            test_sub_pitch2 = subsurf2.get_pitch()
            self.assertEqual(sub_pitch2, test_sub_pitch2)