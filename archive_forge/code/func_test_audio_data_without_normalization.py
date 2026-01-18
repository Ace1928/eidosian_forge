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
def test_audio_data_without_normalization(self):
    max_int16 = numpy.iinfo(numpy.int16).max
    for scale in [1, 0.5, 0.2]:
        test_tone = get_test_tone(scale)
        test_tone_max_abs = numpy.max(numpy.abs(test_tone))
        expected_max_value = int(max_int16 * test_tone_max_abs)
        audio = display.Audio(test_tone, rate=44100, normalize=False)
        actual_max_value = numpy.max(numpy.abs(read_wav(audio.data)))
        assert actual_max_value == expected_max_value