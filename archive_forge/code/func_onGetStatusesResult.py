from yowsup.layers import  YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity, ResultIqProtocolEntity
def onGetStatusesResult(self, resultNode, originIqRequestEntity):
    self.toUpper(ResultStatusesIqProtocolEntity.fromProtocolTreeNode(resultNode))