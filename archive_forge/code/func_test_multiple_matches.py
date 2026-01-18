import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_multiple_matches(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <input py:match="form//input" py:attrs="select(\'@*\')"\n                 value="${values[str(select(\'@name\'))]}" />\n          <form><p py:for="field in fields">\n            <label>${field.capitalize()}</label>\n            <input type="text" name="${field}" />\n          </p></form>\n        </html>')
    fields = ['hello_%s' % i for i in range(5)]
    values = dict([('hello_%s' % i, i) for i in range(5)])
    self.assertEqual('<html>\n          <form><p>\n            <label>Hello_0</label>\n            <input value="0" type="text" name="hello_0"/>\n          </p><p>\n            <label>Hello_1</label>\n            <input value="1" type="text" name="hello_1"/>\n          </p><p>\n            <label>Hello_2</label>\n            <input value="2" type="text" name="hello_2"/>\n          </p><p>\n            <label>Hello_3</label>\n            <input value="3" type="text" name="hello_3"/>\n          </p><p>\n            <label>Hello_4</label>\n            <input value="4" type="text" name="hello_4"/>\n          </p></form>\n        </html>', tmpl.generate(fields=fields, values=values).render(encoding=None))