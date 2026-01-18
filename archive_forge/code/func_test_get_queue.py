import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_queue(self):
    """Ensure Channel.get_queue() returns any queued Sound."""
    channel = mixer.Channel(0)
    frequency, format, channels = mixer.get_init()
    sound_length_in_ms = 200
    sound_length_in_ms_2 = 400
    bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
    sound1 = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
    sound2 = mixer.Sound(b'\x00' * int(sound_length_in_ms_2 * bytes_per_ms))
    channel.play(sound1)
    channel.queue(sound2)
    self.assertEqual(channel.get_queue().get_length(), sound2.get_length())
    pygame.time.wait(sound_length_in_ms + 100)
    self.assertEqual(channel.get_sound().get_length(), sound2.get_length())
    pygame.time.wait(sound_length_in_ms_2 + 100)
    self.assertIsNone(channel.get_queue())