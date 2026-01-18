from tempfile import NamedTemporaryFile, mkdtemp
from os.path import split, join as pjoin, dirname
import pathlib
from unittest import TestCase, mock
import struct
import wave
from io import BytesIO
import pytest
from IPython.lib import display
from IPython.testing.decorators import skipif_not_numpy
def test_instantiation_FileLink():
    """FileLink: Test class can be instantiated"""
    fl = display.FileLink('example.txt')
    fl = display.FileLink(pathlib.PurePath('example.txt'))