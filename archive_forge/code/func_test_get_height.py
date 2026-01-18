from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_get_height(self):
    f = pygame_font.Font(None, 20)
    height = f.get_height()
    self.assertTrue(isinstance(height, int))
    self.assertTrue(height > 0)
    s = f.render('X', False, (255, 255, 255))
    self.assertTrue(s.get_size()[1] == height)