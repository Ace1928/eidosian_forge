from yowsup.layers import  YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity, ResultIqProtocolEntity
def onGetPictureError(self, errorNode, originalIqRequestEntity):
    self.toUpper(ErrorIqProtocolEntity.fromProtocolTreeNode(errorNode))