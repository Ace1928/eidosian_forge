from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onLeaveGroupSuccess(self, node, originalIqEntity):
    logger.info('Group leave success')
    self.toUpper(SuccessLeaveGroupsIqProtocolEntity.fromProtocolTreeNode(node))