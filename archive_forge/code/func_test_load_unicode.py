import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_load_unicode(self):
    """test non-ASCII unicode path"""
    import shutil
    ep = example_path('data')
    temp_file = os.path.join(ep, '你好.wav')
    org_file = os.path.join(ep, 'house_lo.wav')
    try:
        with open(temp_file, 'w') as f:
            pass
        os.remove(temp_file)
    except OSError:
        raise unittest.SkipTest('the path cannot be opened')
    shutil.copy(org_file, temp_file)
    try:
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.load(org_file)
    finally:
        os.remove(temp_file)