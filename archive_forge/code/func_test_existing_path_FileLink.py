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
def test_existing_path_FileLink():
    """FileLink: Calling _repr_html_ functions as expected on existing filepath
    """
    tf = NamedTemporaryFile()
    fl = display.FileLink(tf.name)
    actual = fl._repr_html_()
    expected = "<a href='%s' target='_blank'>%s</a><br>" % (tf.name, tf.name)
    assert actual == expected