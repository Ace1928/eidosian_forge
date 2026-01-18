import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
def test_load_unicode_path_0(self):
    u = example_path('data/alien1.png')
    im = imageext.load_extended(u)