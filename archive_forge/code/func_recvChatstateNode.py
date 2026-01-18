from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import * 
def recvChatstateNode(self, node):
    self.toUpper(IncomingChatstateProtocolEntity.fromProtocolTreeNode(node))