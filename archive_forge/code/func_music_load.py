import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def music_load(self, format):
    data_fname = example_path('data')
    path = os.path.join(data_fname, f'house_lo.{format}')
    if os.sep == '\\':
        path = path.replace('\\', '\\\\')
    umusfn = str(path)
    bmusfn = umusfn.encode()
    pygame.mixer.music.load(umusfn)
    pygame.mixer.music.load(bmusfn)