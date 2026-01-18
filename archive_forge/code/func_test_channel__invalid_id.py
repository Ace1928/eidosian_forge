import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_channel__invalid_id(self):
    """Ensure exception for Channel() creation with an invalid id."""
    with self.assertRaises(IndexError):
        mixer.Channel(-1)