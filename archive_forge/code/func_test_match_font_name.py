from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_match_font_name(self):
    """That match_font accepts names of various types"""
    font = pygame_font.get_fonts()[0]
    font_path = pygame_font.match_font(font)
    self.assertIsNotNone(font_path)
    font_b = font.encode()
    not_a_font = 'thisisnotafont'
    not_a_font_b = b'thisisnotafont'
    good_font_names = [font_b, ','.join([not_a_font, font, not_a_font]), [not_a_font, font, not_a_font], (name for name in [not_a_font, font, not_a_font]), b','.join([not_a_font_b, font_b, not_a_font_b]), [not_a_font_b, font_b, not_a_font_b], [font, not_a_font, font_b, not_a_font_b]]
    for font_name in good_font_names:
        self.assertEqual(pygame_font.match_font(font_name), font_path, font_name)