import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_get_caption(self):
    screen = display.set_mode((100, 100))
    self.assertEqual(display.get_caption()[0], self.default_caption)