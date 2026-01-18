from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_receipts.protocolentities import IncomingReceiptProtocolEntity
from yowsup.layers.axolotl.protocolentities.iq_keys_get_result import ResultGetKeysIqProtocolEntity
def getRetryJid(self):
    return self.getParticipant() or self.getFrom()