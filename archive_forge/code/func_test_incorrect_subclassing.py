import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_incorrect_subclassing(self):

    class IncorrectSuclass(mixer.Sound):

        def __init__(self):
            pass
    incorrect = IncorrectSuclass()
    self.assertRaises(RuntimeError, incorrect.get_volume)