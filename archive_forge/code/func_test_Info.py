import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_Info(self):
    inf = pygame.display.Info()
    self.assertNotEqual(inf.current_h, -1)
    self.assertNotEqual(inf.current_w, -1)
    screen = pygame.display.set_mode((128, 128))
    inf = pygame.display.Info()
    self.assertEqual(inf.current_h, 128)
    self.assertEqual(inf.current_w, 128)