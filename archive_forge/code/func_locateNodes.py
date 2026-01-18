import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def locateNodes(nodeList, key, value, noNesting=1):
    """
    Find subnodes in the given node where the given attribute
    has the given value.
    """
    returnList = []
    if not isinstance(nodeList, type([])):
        return locateNodes(nodeList.childNodes, key, value, noNesting)
    for childNode in nodeList:
        if not hasattr(childNode, 'getAttribute'):
            continue
        if str(childNode.getAttribute(key)) == value:
            returnList.append(childNode)
            if noNesting:
                continue
        returnList.extend(locateNodes(childNode, key, value, noNesting))
    return returnList