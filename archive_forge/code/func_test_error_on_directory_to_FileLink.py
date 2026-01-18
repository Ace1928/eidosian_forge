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
def test_error_on_directory_to_FileLink():
    """FileLink: Raises error when passed directory
    """
    td = mkdtemp()
    pytest.raises(ValueError, display.FileLink, td)