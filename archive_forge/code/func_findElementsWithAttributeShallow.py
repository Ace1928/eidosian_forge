import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def findElementsWithAttributeShallow(parent, attribute):
    """
    Return an iterable of the elements which are direct children of C{parent}
    and which have the C{attribute} attribute.
    """
    return findNodesShallow(parent, lambda n: getattr(n, 'tagName', None) is not None and n.hasAttribute(attribute))