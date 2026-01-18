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
def test_magic(f, magic_hexes):
    """Tests a given file to see if the magic hex matches."""
    data = f.read(len(magic_hexes))
    if len(data) != len(magic_hexes):
        return 0
    for i, magic_hex in enumerate(magic_hexes):
        if magic_hex != data[i]:
            return 0
    return 1