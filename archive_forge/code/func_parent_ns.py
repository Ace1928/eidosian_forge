import io
import sys
import six
import types
from six import StringIO
from io import BytesIO
from lxml import etree
from ncclient import NCClientError
def parent_ns(node):
    if node.prefix:
        return node.nsmap[node.prefix]
    return None