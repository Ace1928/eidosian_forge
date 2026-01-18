import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_caption_unicode(self):
    TEST_CAPTION = '台'
    display.set_caption(TEST_CAPTION)
    self.assertEqual(display.get_caption()[0], TEST_CAPTION)