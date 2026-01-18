from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onRemoveParticipantsSuccess(self, node, originalIqEntity):
    logger.info('Group remove participants success')
    self.toUpper(SuccessRemoveParticipantsIqProtocolEntity.fromProtocolTreeNode(node))