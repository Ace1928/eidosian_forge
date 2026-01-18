import unittest
import pygame
def test_get_default_output_id(self):
    c = pygame.midi.get_default_output_id()
    self.assertIsInstance(c, int)
    self.assertTrue(c >= -1)
    pygame.midi.quit()
    self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)