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
def handleSenderKeyDistributionMessage(self, senderKeyDistributionMessage, participantId):
    groupId = senderKeyDistributionMessage.group_id
    self.manager.group_create_session(groupid=groupId, participantid=participantId, skmsgdata=senderKeyDistributionMessage.axolotl_sender_key_distribution_message)