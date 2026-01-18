from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_font_set_script(self):
    if pygame_font.__name__ == 'pygame.ftfont':
        return
    font = pygame_font.Font(None, 16)
    ttf_version = pygame_font.get_sdl_ttf_version()
    if ttf_version >= (2, 20, 0):
        self.assertRaises(TypeError, pygame.font.Font.set_script)
        self.assertRaises(TypeError, pygame.font.Font.set_script, font)
        self.assertRaises(TypeError, pygame.font.Font.set_script, 'hey', 'Deva')
        self.assertRaises(TypeError, font.set_script, 1)
        self.assertRaises(TypeError, font.set_script, ['D', 'e', 'v', 'a'])
        self.assertRaises(ValueError, font.set_script, 'too long by far')
        self.assertRaises(ValueError, font.set_script, '')
        self.assertRaises(ValueError, font.set_script, 'a')
        font.set_script('Deva')
    else:
        self.assertRaises(pygame.error, font.set_script, 'Deva')