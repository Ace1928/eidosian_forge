import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_display_surface_quit(self):
    """Font.render_to() on a closed display surface"""
    null_surface = pygame.Surface.__new__(pygame.Surface)
    f = self._TEST_FONTS['sans']
    self.assertRaises(pygame.error, f.render_to, null_surface, (0, 0), 'Crash!', size=12)