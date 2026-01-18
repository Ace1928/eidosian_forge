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
def test_existing_path_FileLinks_alt_formatter():
    """FileLinks: Calling _repr_html_ functions as expected w/ an alt formatter
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)

    def fake_formatter(dirname, fnames, included_suffixes):
        return ['hello', 'world']
    fl = display.FileLinks(td, notebook_display_formatter=fake_formatter)
    actual = fl._repr_html_()
    actual = actual.split('\n')
    actual.sort()
    expected = ['hello', 'world']
    expected.sort()
    assert actual == expected