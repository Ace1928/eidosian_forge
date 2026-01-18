from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_load_from_file_unicode_0(self):
    """ASCII string as a unicode object"""
    self._load_unicode('temp_file.ttf')