from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onAddParticipantsFailed(self, node, originalIqEntity):
    logger.error('Group add participants failed')
    self.toUpper(FailureAddParticipantsIqProtocolEntity.fromProtocolTreeNode(node))