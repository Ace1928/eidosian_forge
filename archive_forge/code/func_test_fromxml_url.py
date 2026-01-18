from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_fromxml_url():
    try:
        url = 'http://raw.githubusercontent.com/petl-developers/petl/master/petl/test/resources/test.xml'
        urlopen(url)
        import pkg_resources
        filename = pkg_resources.resource_filename('petl', 'test/resources/test.xml')
    except Exception as e:
        pytest.skip('SKIP test_fromxml_url: %s' % e)
    else:
        actual = fromxml(url, 'pydev_property', {'name': ('.', 'name'), 'prop': '.'})
        assert nrows(actual) > 0
        expect = fromxml(filename, 'pydev_property', {'name': ('.', 'name'), 'prop': '.'})
        ieq(expect, actual)