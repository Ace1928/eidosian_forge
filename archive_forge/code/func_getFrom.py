from yowsup.structs import ProtocolEntity, ProtocolTreeNode
def getFrom(self, full=True):
    return self._from if full else self._from.split('@')[0]