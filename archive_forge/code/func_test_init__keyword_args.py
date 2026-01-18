import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_init__keyword_args(self):
    mixer.init(**CONFIG)
    mixer_conf = mixer.get_init()
    self.assertEqual(mixer_conf[0], CONFIG['frequency'])
    self.assertEqual(abs(mixer_conf[1]), abs(CONFIG['size']))
    self.assertGreaterEqual(mixer_conf[2], CONFIG['channels'])