from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_receipts.protocolentities import OutgoingReceiptProtocolEntity
from yowsup.layers.axolotl.protocolentities.iq_keys_get_result import ResultGetKeysIqProtocolEntity
@retry_timestamp.setter
def retry_timestamp(self, value):
    self._retry_timestamp = value