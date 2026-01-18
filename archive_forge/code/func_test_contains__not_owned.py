import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_contains__not_owned(self):
    """Ensures contains works when the clipboard is not owned
        by the pygame application.
        """
    self._skip_if_clipboard_owned()
    DATA_TYPE = 'test_contains__not_owned'
    contains = scrap.contains(DATA_TYPE)
    self.assertFalse(contains)