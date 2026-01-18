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
def simulate_numpy_not_installed():
    try:
        import numpy
        return mock.patch('numpy.array', mock.MagicMock(side_effect=ImportError))
    except ModuleNotFoundError:
        return lambda x: x