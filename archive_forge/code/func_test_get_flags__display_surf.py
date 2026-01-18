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
def test_get_flags__display_surf(self):
    pygame.display.init()
    try:
        screen_surf = pygame.display.set_mode((600, 400), flags=0)
        self.assertFalse(screen_surf.get_flags() & pygame.FULLSCREEN)
        screen_surf = pygame.display.set_mode((600, 400), flags=pygame.FULLSCREEN)
        self.assertTrue(screen_surf.get_flags() & pygame.FULLSCREEN)
        screen_surf = pygame.display.set_mode((600, 400), flags=0)
        self.assertFalse(screen_surf.get_flags() & pygame.NOFRAME)
        screen_surf = pygame.display.set_mode((600, 400), flags=pygame.NOFRAME)
        self.assertTrue(screen_surf.get_flags() & pygame.NOFRAME)
        screen_surf = pygame.display.set_mode((600, 400), flags=0)
        self.assertFalse(screen_surf.get_flags() & pygame.RESIZABLE)
        screen_surf = pygame.display.set_mode((600, 400), flags=pygame.RESIZABLE)
        self.assertTrue(screen_surf.get_flags() & pygame.RESIZABLE)
        screen_surf = pygame.display.set_mode((600, 400), flags=0)
        if not screen_surf.get_flags() & pygame.OPENGL:
            self.assertFalse(screen_surf.get_flags() & pygame.OPENGL)
        try:
            pygame.display.set_mode((200, 200), pygame.OPENGL, 32)
        except pygame.error:
            pass
        else:
            self.assertTrue(screen_surf.get_flags() & pygame.OPENGL)
    finally:
        pygame.display.quit()