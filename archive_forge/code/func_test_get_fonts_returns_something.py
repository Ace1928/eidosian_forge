from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_get_fonts_returns_something(self):
    fnts = pygame_font.get_fonts()
    self.assertTrue(fnts)