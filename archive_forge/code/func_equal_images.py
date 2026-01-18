from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def equal_images(s1, s2):
    size = s1.get_size()
    if s2.get_size() != size:
        return False
    w, h = size
    for x in range(w):
        for y in range(h):
            if s1.get_at((x, y)) != s2.get_at((x, y)):
                return False
    return True