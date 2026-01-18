from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_load_default_font_filename(self):
    f = pygame_font.Font(pygame_font.get_default_font(), 20)