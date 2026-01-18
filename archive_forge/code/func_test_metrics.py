from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_metrics(self):
    f = pygame_font.Font(None, 20)
    um = f.metrics('.')
    bm = f.metrics(b'.')
    self.assertEqual(len(um), 1)
    self.assertEqual(len(bm), 1)
    self.assertIsNotNone(um[0])
    self.assertEqual(um, bm)
    u = 'K'
    b = u.encode('UTF-16')[2:]
    bm = f.metrics(b)
    self.assertEqual(len(bm), 2)
    try:
        um = f.metrics(u)
    except pygame.error:
        pass
    else:
        self.assertEqual(len(um), 1)
        self.assertNotEqual(bm[0], um[0])
        self.assertNotEqual(bm[1], um[0])
    u = '𓀀'
    bm = f.metrics(u)
    self.assertEqual(len(bm), 1)
    self.assertIsNone(bm[0])
    return
    self.fail()