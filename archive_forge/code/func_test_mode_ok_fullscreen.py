import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_mode_ok_fullscreen(self):
    modes = pygame.display.list_modes()
    if modes != -1:
        size = modes[0]
        self.assertNotEqual(pygame.display.mode_ok(size, flags=pygame.FULLSCREEN), 0)