from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def nodesEqual(self, node1, node2):
    if not node1.nameTuple == node2.nameTuple:
        return False
    if not node1.attributes == node2.attributes:
        return False
    return True