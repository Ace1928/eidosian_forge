def protocolTreeNodeToBytes(self, node):
    outBytes = [0]
    self.writeInternal(node, outBytes)
    return outBytes