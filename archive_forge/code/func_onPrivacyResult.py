from yowsup.layers import  YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity, ResultIqProtocolEntity
def onPrivacyResult(self, resultNode, originIqRequestEntity):
    self.toUpper(ResultPrivacyIqProtocolEntity.fromProtocolTreeNode(resultNode))