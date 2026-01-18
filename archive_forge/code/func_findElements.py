import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def findElements(parent, matcher):
    """
    Return an iterable of the elements which are children of C{parent} for
    which the predicate C{matcher} returns true.
    """
    return findNodes(parent, lambda n, matcher=matcher: getattr(n, 'tagName', None) is not None and matcher(n))