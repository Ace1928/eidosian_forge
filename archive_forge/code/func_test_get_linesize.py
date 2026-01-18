from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_get_linesize(self):
    f = pygame_font.Font(None, 20)
    linesize = f.get_linesize()
    self.assertTrue(isinstance(linesize, int))
    self.assertTrue(linesize > 0)