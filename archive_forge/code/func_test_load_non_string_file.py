import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
def test_load_non_string_file(self):
    self.assertRaises(TypeError, imageext.load_extended, [])