import io
import sys
import six
import types
from six import StringIO
from io import BytesIO
from lxml import etree
from ncclient import NCClientError
def parse_root(raw):
    """Efficiently parses the root element of a *raw* XML document, returning a tuple of its qualified name and attribute dictionary."""
    if sys.version < '3':
        fp = StringIO(raw)
    else:
        fp = BytesIO(raw.encode('UTF-8'))
    for event, element in etree.iterparse(fp, events=('start',)):
        return (element.tag, element.attrib)