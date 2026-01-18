import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def gatherTextNodes(iNode, dounescape=0, joinWith=''):
    """Visit each child node and collect its text data, if any, into a string.
    For example::
        >>> doc=microdom.parseString('<a>1<b>2<c>3</c>4</b></a>')
        >>> gatherTextNodes(doc.documentElement)
        '1234'
    With dounescape=1, also convert entities back into normal characters.
    @return: the gathered nodes as a single string
    @rtype: str"""
    gathered = []
    gathered_append = gathered.append
    slice = [iNode]
    while len(slice) > 0:
        c = slice.pop(0)
        if hasattr(c, 'nodeValue') and c.nodeValue is not None:
            if dounescape:
                val = unescape(c.nodeValue)
            else:
                val = c.nodeValue
            gathered_append(val)
        slice[:0] = c.childNodes
    return joinWith.join(gathered)