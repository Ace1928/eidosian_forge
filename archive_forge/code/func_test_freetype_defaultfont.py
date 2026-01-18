import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_defaultfont(self):
    font = ft.Font(None)
    self.assertEqual(font.name, 'FreeSans')