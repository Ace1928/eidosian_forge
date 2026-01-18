import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_busy__active(self):
    """Ensure an active channel's busy state is correct."""
    channel = mixer.Channel(0)
    sound = mixer.Sound(example_path('data/house_lo.wav'))
    channel.play(sound)
    self.assertTrue(channel.get_busy())