import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_sdl_mixer_version__invalid_args_kwargs(self):
    """Ensures get_sdl_mixer_version handles invalid args and kwargs."""
    invalid_bool = InvalidBool()
    with self.assertRaises(TypeError):
        version = pygame.mixer.get_sdl_mixer_version(invalid_bool)
    with self.assertRaises(TypeError):
        version = pygame.mixer.get_sdl_mixer_version(linked=invalid_bool)