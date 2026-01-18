from .layer_base import AxolotlBaseLayer
from yowsup.layers.protocol_receipts.protocolentities import OutgoingReceiptProtocolEntity
from yowsup.layers.protocol_messages.proto.e2e_pb2 import *
from yowsup.layers.axolotl.protocolentities import *
from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_messages.protocolentities.proto import ProtoProtocolEntity
from yowsup.layers.axolotl.props import PROP_IDENTITY_AUTOTRUST
from yowsup.axolotl import exceptions
from axolotl.untrustedidentityexception import UntrustedIdentityException
import logging
def send_retry(self, message_node, registration_id):
    message_id = message_node['id']
    if message_id in self._retries:
        count = self._retries[message_id]
        count += 1
    else:
        count = 1
    self._retries[message_id] = count
    retry = RetryOutgoingReceiptProtocolEntity.fromMessageNode(message_node, registration_id)
    retry.count = count
    self.toLower(retry.toProtocolTreeNode())