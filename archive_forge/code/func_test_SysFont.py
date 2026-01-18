from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_SysFont(self):
    fonts = pygame_font.get_fonts()
    if 'arial' in fonts:
        font_name = 'arial'
    else:
        font_name = sorted(fonts)[0]
    o = pygame_font.SysFont(font_name, 20)
    self.assertTrue(isinstance(o, pygame_font.FontType))
    o = pygame_font.SysFont(font_name, 20, italic=True)
    self.assertTrue(isinstance(o, pygame_font.FontType))
    o = pygame_font.SysFont(font_name, 20, bold=True)
    self.assertTrue(isinstance(o, pygame_font.FontType))
    o = pygame_font.SysFont('thisisnotafont', 20)
    self.assertTrue(isinstance(o, pygame_font.FontType))