import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def test_load_unicode_path(self):
    import shutil
    orig = example_path('data/asprite.bmp')
    temp = os.path.join(example_path('data'), '你好.bmp')
    shutil.copy(orig, temp)
    try:
        im = pygame.image.load(temp)
    finally:
        os.remove(temp)