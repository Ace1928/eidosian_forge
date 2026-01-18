from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_fromxml():
    f = NamedTemporaryFile(delete=False, mode='wt')
    data = '<table>\n        <tr>\n            <td>foo</td><td>bar</td>\n        </tr>\n        <tr>\n            <td>a</td><td>1</td>\n        </tr>\n        <tr>\n            <td>b</td><td>2</td>\n        </tr>\n        <tr>\n            <td>c</td><td>2</td>\n        </tr>\n      </table>'
    f.write(data)
    f.close()
    actual = fromxml(f.name, 'tr', 'td')
    expect = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '2'))
    ieq(expect, actual)
    ieq(expect, actual)