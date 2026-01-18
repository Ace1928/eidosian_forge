from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def elementInScope(self, target, variant=None):
    exactNode = hasattr(target, 'nameTuple')
    if not exactNode:
        if isinstance(target, text_type):
            target = (namespaces['html'], target)
        assert isinstance(target, tuple)
    listElements, invert = listElementsMap[variant]
    for node in reversed(self.openElements):
        if exactNode and node == target:
            return True
        elif not exactNode and node.nameTuple == target:
            return True
        elif invert ^ (node.nameTuple in listElements):
            return False
    assert False