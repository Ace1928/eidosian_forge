import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_issue_243(self):
    """Issue Y: trailing space ignored in boundary calculation"""
    font = self._TEST_FONTS['fixed']
    r1 = font.get_rect(' ', size=64)
    self.assertTrue(r1.width > 1)
    r2 = font.get_rect('  ', size=64)
    self.assertEqual(r2.width, 2 * r1.width)