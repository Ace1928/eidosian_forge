from yowsup.layers import  YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity, ResultIqProtocolEntity
def onGetPictureResult(self, resultNode, originalIqRequestEntity):
    self.toUpper(ResultGetPictureIqProtocolEntity.fromProtocolTreeNode(resultNode))