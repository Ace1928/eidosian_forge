from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_issue_3144(self):
    fpath = os.path.join(FONTDIR, 'PlayfairDisplaySemibold.ttf')
    for size in (60, 40, 10, 20, 70, 45, 50, 10):
        font = pygame_font.Font(fpath, size)
        font.render('WHERE', True, 'black')