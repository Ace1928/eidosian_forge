from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def insertElementTable(self, token):
    """Create an element and insert it into the tree"""
    element = self.createElement(token)
    if self.openElements[-1].name not in tableInsertModeElements:
        return self.insertElementNormal(token)
    else:
        parent, insertBefore = self.getTableMisnestedNodePosition()
        if insertBefore is None:
            parent.appendChild(element)
        else:
            parent.insertBefore(element, insertBefore)
        self.openElements.append(element)
    return element