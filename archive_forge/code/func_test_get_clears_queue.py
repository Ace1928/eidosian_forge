import collections
import time
import unittest
import os
import pygame
def test_get_clears_queue(self):
    """Ensure get() clears the event queue after a call"""
    pygame.event.get()
    self.assertEqual(pygame.event.get(), [])