from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
def onLastSeenError(self, protocolTreeNode, lastSeenEntity):
    self.toUpper(ErrorIqProtocolEntity.fromProtocolTreeNode(protocolTreeNode))