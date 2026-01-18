import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_sound(self):
    """Ensure Channel.get_sound() returns the currently playing Sound."""
    channel = mixer.Channel(0)
    sound = mixer.Sound(example_path('data/house_lo.wav'))
    channel.play(sound)
    got_sound = channel.get_sound()
    self.assertEqual(got_sound, sound)
    channel.stop()
    got_sound = channel.get_sound()
    self.assertIsNone(got_sound)