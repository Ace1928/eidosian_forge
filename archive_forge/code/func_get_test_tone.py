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
@skipif_not_numpy
def get_test_tone(scale=1):
    return numpy.sin(2 * numpy.pi * 440 * numpy.linspace(0, 1, 44100)) * scale