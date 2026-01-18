import unittest
import os
import platform
import warnings
import pygame
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER', '') == 'dummy', 'Cursors not supported on headless test machines')
def test_get_cursor(self):
    """Ensures get_cursor works correctly."""
    with self.assertRaises(pygame.error):
        pygame.display.quit()
        pygame.mouse.get_cursor()
    pygame.display.init()
    size = (8, 8)
    hotspot = (0, 0)
    xormask = (0, 96, 120, 126, 112, 96, 0, 0)
    andmask = (224, 240, 254, 255, 254, 240, 96, 0)
    expected_length = 4
    expected_cursor = pygame.cursors.Cursor(size, hotspot, xormask, andmask)
    pygame.mouse.set_cursor(expected_cursor)
    try:
        cursor = pygame.mouse.get_cursor()
        self.assertIsInstance(cursor, pygame.cursors.Cursor)
        self.assertEqual(len(cursor), expected_length)
        for info in cursor:
            self.assertIsInstance(info, tuple)
        pygame.mouse.set_cursor(size, hotspot, xormask, andmask)
        self.assertEqual(pygame.mouse.get_cursor(), expected_cursor)
    except pygame.error:
        with self.assertRaises(pygame.error):
            pygame.mouse.get_cursor()