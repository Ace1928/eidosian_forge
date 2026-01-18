import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def getAndClear(node, nodeId):
    """Get a node with the specified C{nodeId} as any of the C{class},
    C{id} or C{pattern} attributes. If there is no such node, raise
    L{NodeLookupError}. Remove all child nodes before returning.
    """
    result = get(node, nodeId)
    if result:
        clearNode(result)
    return result