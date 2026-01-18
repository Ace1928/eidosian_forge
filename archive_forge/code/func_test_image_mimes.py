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
def test_image_mimes():
    fmt = get_ipython().display_formatter.format
    for format in display.Image._ACCEPTABLE_EMBEDDINGS:
        mime = display.Image._MIMETYPES[format]
        img = display.Image(b'garbage', format=format)
        data, metadata = fmt(img)
        assert sorted(data) == sorted([mime, 'text/plain'])