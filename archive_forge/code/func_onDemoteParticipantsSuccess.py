from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onDemoteParticipantsSuccess(self, node, originalIqEntity):
    logger.info('Group demote participants success')
    self.toUpper(ResultIqProtocolEntity.fromProtocolTreeNode(node))