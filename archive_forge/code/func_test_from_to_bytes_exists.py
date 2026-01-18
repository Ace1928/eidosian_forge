import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def test_from_to_bytes_exists(self):
    getattr(pygame.image, 'frombytes')
    getattr(pygame.image, 'tobytes')