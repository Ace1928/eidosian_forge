from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
def recvReceiptNode(self, node):
    self.toUpper(IncomingReceiptProtocolEntity.fromProtocolTreeNode(node))