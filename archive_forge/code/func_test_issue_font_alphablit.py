from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_issue_font_alphablit(self):
    """Check that blitting anti-aliased text doesn't
        change the background blue"""
    pygame.display.set_mode((600, 400))
    font = pygame_font.Font(None, 24)
    color, text, center, pos = ((160, 200, 250), 'Music', (190, 170), 'midright')
    img1 = font.render(text, True, color)
    img = pygame.Surface(img1.get_size(), depth=32)
    pre_blit_corner_pixel = img.get_at((0, 0))
    img.blit(img1, (0, 0))
    post_blit_corner_pixel = img.get_at((0, 0))
    self.assertEqual(pre_blit_corner_pixel, post_blit_corner_pixel)