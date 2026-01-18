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
def test_audio_data_normalization(self):
    expected_max_value = numpy.iinfo(numpy.int16).max
    for scale in [1, 0.5, 2]:
        audio = display.Audio(get_test_tone(scale), rate=44100)
        actual_max_value = numpy.max(numpy.abs(read_wav(audio.data)))
        assert actual_max_value == expected_max_value