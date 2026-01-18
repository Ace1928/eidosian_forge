import unittest
import platform
@unittest.skipIf('Windows' not in platform.platform(), 'Not windows we skip.')
def test_initsysfonts_win32(self):
    import pygame.sysfont
    self.assertTrue(len(pygame.sysfont.get_fonts()) > 10)