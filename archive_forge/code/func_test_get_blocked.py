import collections
import time
import unittest
import os
import pygame
def test_get_blocked(self):
    """Ensure an event's blocked state can be retrieved."""
    pygame.event.set_allowed(None)
    for etype in EVENT_TYPES:
        blocked = pygame.event.get_blocked(etype)
        self.assertFalse(blocked)
    pygame.event.set_blocked(None)
    for etype in EVENT_TYPES:
        blocked = pygame.event.get_blocked(etype)
        self.assertTrue(blocked)