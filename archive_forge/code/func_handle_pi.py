from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
def handle_pi(self, data):
    if data.endswith('?'):
        data = data[:-1]
    try:
        target, data = data.split(None, 1)
    except ValueError:
        target = data
        data = ''
    self._enqueue(PI, (target.strip(), data.strip()))