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
def test_embed_svg_url():
    import gzip
    from io import BytesIO
    svg_data = b'<svg><circle x="0" y="0" r="1"/></svg>'
    url = 'http://test.com/circle.svg'
    gzip_svg = BytesIO()
    with gzip.open(gzip_svg, 'wb') as fp:
        fp.write(svg_data)
    gzip_svg = gzip_svg.getvalue()

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
    with mock.patch('urllib.request.urlopen', side_effect=mocked_urlopen):
        svg = display.SVG(url=url)
        assert svg._repr_svg_().startswith('<svg') is True
        svg = display.SVG(url=url + 'z')
        assert svg._repr_svg_().startswith('<svg') is True