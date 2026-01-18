from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_match_font_bold(self):
    fonts = pygame_font.get_fonts()
    self.assertTrue(any((pygame_font.match_font(font, bold=True) for font in fonts)))