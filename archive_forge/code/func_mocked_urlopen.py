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
def mocked_urlopen(*args, **kwargs):

    class MockResponse:

        def __init__(self, svg):
            self._svg_data = svg
            self.headers = {'content-type': 'image/svg+xml'}

        def read(self):
            return self._svg_data
    if args[0] == url:
        return MockResponse(svg_data)
    elif args[0] == url + 'z':
        ret = MockResponse(gzip_svg)
        ret.headers['content-encoding'] = 'gzip'
        return ret
    return MockResponse(None)