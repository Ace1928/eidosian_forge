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
def test_existing_path_FileLinks_repr():
    """FileLinks: Calling repr() functions as expected on existing directory """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)
    fl = display.FileLinks(td)
    actual = repr(fl)
    actual = actual.split('\n')
    actual.sort()
    expected = ['%s/' % td, '  %s' % split(tf1.name)[1], '  %s' % split(tf2.name)[1]]
    expected.sort()
    assert actual == expected