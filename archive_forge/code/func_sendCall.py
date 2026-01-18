from yowsup.layers import YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_acks.protocolentities import OutgoingAckProtocolEntity
from yowsup.layers.protocol_receipts.protocolentities import OutgoingReceiptProtocolEntity
def sendCall(self, entity):
    if entity.getTag() == 'call':
        self.toLower(entity.toProtocolTreeNode())