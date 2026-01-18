import unittest
import os
import pygame
from pygame._sdl2 import touch
from pygame.tests.test_utils import question
def test_get_device__invalid(self):
    self.assertRaises(pygame.error, touch.get_device, -1234)
    self.assertRaises(TypeError, touch.get_device, 'test')