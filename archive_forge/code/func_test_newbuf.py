import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
@unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
@unittest.skipIf(IS_PYPY, 'pypy no likey')
def test_newbuf(self):
    from pygame.tests.test_utils import buftools
    Exporter = buftools.Exporter
    font = self._TEST_FONTS['sans']
    srect = font.get_rect('Hi', size=12)
    for format in ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '<h', '>h', '=h', '@h', '!h', '1h', '=1h']:
        newbuf = Exporter(srect.size, format=format)
        rrect = font.render_raw_to(newbuf, 'Hi', size=12)
        self.assertEqual(rrect, srect)
    for format in ['f', 'd', '2h', '?', 'hh']:
        newbuf = Exporter(srect.size, format=format, itemsize=4)
        self.assertRaises(ValueError, font.render_raw_to, newbuf, 'Hi', size=12)