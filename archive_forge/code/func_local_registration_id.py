from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_receipts.protocolentities import OutgoingReceiptProtocolEntity
from yowsup.layers.axolotl.protocolentities.iq_keys_get_result import ResultGetKeysIqProtocolEntity
@local_registration_id.setter
def local_registration_id(self, value):
    self._local_registration_id = value