import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def smoothscale_low_bpp():
    starting_surface = pygame.Surface((20, 20), depth=12)
    smoothscaled_surface = pygame.transform.smoothscale(starting_surface, (10, 10))