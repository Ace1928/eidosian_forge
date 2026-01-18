from yowsup.structs import ProtocolEntity, ProtocolTreeNode
def getErrorType(self):
    for k in self.data.keys():
        if k in self.__class__.TYPES:
            return k