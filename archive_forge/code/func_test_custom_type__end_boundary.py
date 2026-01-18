import collections
import time
import unittest
import os
import pygame
def test_custom_type__end_boundary(self):
    """Ensure custom_type() raises error when no more custom types.

        The last allowed custom type number should be (pygame.NUMEVENTS - 1).
        """
    last = -1
    start = pygame.event.custom_type() + 1
    for _ in range(start, pygame.NUMEVENTS):
        last = pygame.event.custom_type()
    self.assertEqual(last, pygame.NUMEVENTS - 1)
    with self.assertRaises(pygame.error):
        pygame.event.custom_type()