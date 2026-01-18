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
def test_existing_path_FileLinks():
    """FileLinks: Calling _repr_html_ functions as expected on existing dir
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)
    fl = display.FileLinks(td)
    actual = fl._repr_html_()
    actual = actual.split('\n')
    actual.sort()
    expected = ['%s/<br>' % td, "&nbsp;&nbsp;<a href='%s' target='_blank'>%s</a><br>" % (tf2.name.replace('\\', '/'), split(tf2.name)[1]), "&nbsp;&nbsp;<a href='%s' target='_blank'>%s</a><br>" % (tf1.name.replace('\\', '/'), split(tf1.name)[1])]
    expected.sort()
    assert actual == expected