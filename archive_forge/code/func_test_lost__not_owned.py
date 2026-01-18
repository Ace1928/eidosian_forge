import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_lost__not_owned(self):
    """Ensures lost works when the clipboard is not owned
        by the pygame application.
        """
    self._skip_if_clipboard_owned()
    lost = scrap.lost()
    self.assertTrue(lost)