import unittest
import pygame.constants
def test_kmod__no_bitwise_overlap(self):
    """Ensures certain KMOD constants have no overlapping bits."""
    NO_BITWISE_OVERLAP = ('KMOD_NONE', 'KMOD_LSHIFT', 'KMOD_RSHIFT', 'KMOD_LCTRL', 'KMOD_RCTRL', 'KMOD_LALT', 'KMOD_RALT', 'KMOD_LMETA', 'KMOD_RMETA', 'KMOD_NUM', 'KMOD_CAPS', 'KMOD_MODE')
    kmods = 0
    for name in NO_BITWISE_OVERLAP:
        value = getattr(pygame.constants, name)
        self.assertFalse(kmods & value)
        kmods |= value