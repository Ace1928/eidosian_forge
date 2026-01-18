import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def make_surf(bpp, flags, masks):
    pygame.Surface((10, 10), flags, bpp, masks)