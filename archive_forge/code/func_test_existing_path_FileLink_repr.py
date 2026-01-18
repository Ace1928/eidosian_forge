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
def test_existing_path_FileLink_repr():
    """FileLink: Calling repr() functions as expected on existing filepath
    """
    tf = NamedTemporaryFile()
    fl = display.FileLink(tf.name)
    actual = repr(fl)
    expected = tf.name
    assert actual == expected