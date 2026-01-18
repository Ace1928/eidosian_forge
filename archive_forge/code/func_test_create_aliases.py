import unittest
import platform
def test_create_aliases(self):
    import pygame.sysfont
    pygame.sysfont.initsysfonts()
    pygame.sysfont.create_aliases()
    self.assertTrue(len(pygame.sysfont.Sysalias) > 0)