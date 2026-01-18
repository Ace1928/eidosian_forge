from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_fromxml_2():
    f = NamedTemporaryFile(delete=False, mode='wt')
    data = "<table>\n        <tr>\n            <td v='foo'/><td v='bar'/>\n        </tr>\n        <tr>\n            <td v='a'/><td v='1'/>\n        </tr>\n        <tr>\n            <td v='b'/><td v='2'/>\n        </tr>\n        <tr>\n            <td v='c'/><td v='2'/>\n        </tr>\n      </table>"
    f.write(data)
    f.close()
    actual = fromxml(f.name, 'tr', 'td', 'v')
    expect = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '2'))
    ieq(expect, actual)
    ieq(expect, actual)