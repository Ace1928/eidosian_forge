import unittest
from pygame.tests.test_utils import fixture_path
import pygame
def test_Cursor(self):
    """Ensure that the cursor object parses information properly"""
    c1 = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
    self.assertEqual(c1.data, (pygame.SYSTEM_CURSOR_CROSSHAIR,))
    self.assertEqual(c1.type, 'system')
    c2 = pygame.cursors.Cursor(c1)
    self.assertEqual(c1, c2)
    with self.assertRaises(TypeError):
        pygame.cursors.Cursor(-34002)
    with self.assertRaises(TypeError):
        pygame.cursors.Cursor('a', 'b', 'c', 'd')
    with self.assertRaises(TypeError):
        pygame.cursors.Cursor((2,))
    c3 = pygame.cursors.Cursor((0, 0), pygame.Surface((20, 20)))
    self.assertEqual(c3.data[0], (0, 0))
    self.assertEqual(c3.data[1].get_size(), (20, 20))
    self.assertEqual(c3.type, 'color')
    xormask, andmask = pygame.cursors.compile(pygame.cursors.thickarrow_strings)
    c4 = pygame.cursors.Cursor((24, 24), (0, 0), xormask, andmask)
    self.assertEqual(c4.data, ((24, 24), (0, 0), xormask, andmask))
    self.assertEqual(c4.type, 'bitmap')