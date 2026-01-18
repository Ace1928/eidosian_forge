import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_correct_subclassing(self):

    class CorrectSublass(mixer.Sound):

        def __init__(self, file):
            super().__init__(file=file)
    filename = example_path(os.path.join('data', 'house_lo.wav'))
    correct = CorrectSublass(filename)
    try:
        correct.get_volume()
    except Exception:
        self.fail('This should not raise an exception.')