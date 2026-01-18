import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_get_num_displays(self):
    self.assertGreater(pygame.display.get_num_displays(), 0)