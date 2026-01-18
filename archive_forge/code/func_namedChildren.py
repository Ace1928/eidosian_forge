import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def namedChildren(parent, nodeName):
    """namedChildren(parent, nodeName) -> children (not descendants) of parent
    that have tagName == nodeName
    """
    return [n for n in parent.childNodes if getattr(n, 'tagName', '') == nodeName]