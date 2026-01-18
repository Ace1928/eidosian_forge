import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def do_pixels_alpha(surf):
    pygame.surfarray.pixels_alpha(surf)