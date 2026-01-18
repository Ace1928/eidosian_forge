import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_cache_markup(self):
    loc = (None, -1, -1)
    stream = Stream([(Stream.START, (QName('foo'), Attrs()), loc), (Stream.TEXT, u'&hellip;', loc), (Stream.END, QName('foo'), loc), (Stream.START, (QName('bar'), Attrs()), loc), (Stream.TEXT, Markup('&hellip;'), loc), (Stream.END, QName('bar'), loc)])
    output = stream.render(XMLSerializer, encoding=None, strip_whitespace=False)
    self.assertEqual('<foo>&amp;hellip;</foo><bar>&hellip;</bar>', output)