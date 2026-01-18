import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_fadeout(self):
    """Ensure Channel.fadeout() stops playback after fading out."""
    channel = mixer.Channel(0)
    sound = mixer.Sound(example_path('data/house_lo.wav'))
    channel.play(sound)
    fadeout_time = 1000
    channel.fadeout(fadeout_time)
    pygame.time.wait(fadeout_time + 100)
    self.assertFalse(channel.get_busy())