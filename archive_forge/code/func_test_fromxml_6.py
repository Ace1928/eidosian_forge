from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_fromxml_6():
    data = "<table class='petl'>\n        <thead>\n        <tr>\n        <th>foo</th>\n        <th>bar</th>\n        </tr>\n        </thead>\n        <tbody>\n        <tr>\n        <td>a</td>\n        <td style='text-align: right'>2</td>\n        </tr>\n        <tr>\n        <td>b</td>\n        <td style='text-align: right'>1</td>\n        </tr>\n        <tr>\n        <td>c</td>\n        <td style='text-align: right'>3</td>\n        </tr>\n        </tbody>\n      </table>"
    f = NamedTemporaryFile(delete=False, mode='wt')
    f.write(data)
    f.close()
    actual = fromxml(f.name, './/tr', ('th', 'td'))
    print(look(actual))
    expect = (('foo', 'bar'), ('a', '2'), ('b', '1'), ('c', '3'))
    ieq(expect, actual)
    ieq(expect, actual)