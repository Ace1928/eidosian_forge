import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
def test_image_filename_defaults():
    """test format constraint, and validity of jpeg and png"""
    tpath = ipath.get_ipython_package_dir()
    pytest.raises(ValueError, display.Image, filename=os.path.join(tpath, 'testing/tests/badformat.zip'), embed=True)
    pytest.raises(ValueError, display.Image)
    pytest.raises(ValueError, display.Image, data='this is not an image', format='badformat', embed=True)
    imgfile = os.path.join(tpath, 'core/tests/2x2.png')
    img = display.Image(filename=imgfile)
    assert 'png' == img.format
    assert img._repr_png_() is not None
    img = display.Image(filename=os.path.join(tpath, 'testing/tests/logo.jpg'), embed=False)
    assert 'jpeg' == img.format
    assert img._repr_jpeg_() is None