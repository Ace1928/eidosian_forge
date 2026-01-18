import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def superAppendAttribute(node, key, value):
    if not hasattr(node, 'setAttribute'):
        return
    old = node.getAttribute(key)
    if old:
        node.setAttribute(key, old + '/' + value)
    else:
        node.setAttribute(key, value)
    if node.hasChildNodes():
        for child in node.childNodes:
            superAppendAttribute(child, key, value)