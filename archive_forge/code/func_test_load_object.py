import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_load_object(self):
    """test loading music from file-like objects."""
    formats = ['ogg', 'wav']
    data_fname = example_path('data')
    for f in formats:
        path = os.path.join(data_fname, f'house_lo.{f}')
        if os.sep == '\\':
            path = path.replace('\\', '\\\\')
        bmusfn = path.encode()
        with open(bmusfn, 'rb') as musf:
            pygame.mixer.music.load(musf)