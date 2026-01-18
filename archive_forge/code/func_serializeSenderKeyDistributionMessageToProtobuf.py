from yowsup.layers.protocol_messages.proto.e2e_pb2 import Message
from yowsup.layers.axolotl.protocolentities import *
from yowsup.layers.auth.layer_authentication import YowAuthenticationProtocolLayer
from yowsup.layers.protocol_groups.protocolentities import InfoGroupsIqProtocolEntity, InfoGroupsResultIqProtocolEntity
from axolotl.protocol.whispermessage import WhisperMessage
from yowsup.layers.protocol_messages.protocolentities.message import MessageMetaAttributes
from yowsup.layers.axolotl.protocolentities.iq_keys_get_result import MissingParametersException
from yowsup.axolotl import exceptions
from .layer_base import AxolotlBaseLayer
import logging
def serializeSenderKeyDistributionMessageToProtobuf(self, groupId, senderKeyDistributionMessage, message=None):
    m = message or Message()
    m.sender_key_distribution_message.group_id = groupId
    m.sender_key_distribution_message.axolotl_sender_key_distribution_message = senderKeyDistributionMessage.serialize()
    m.sender_key_distribution_message.axolotl_sender_key_distribution_message = senderKeyDistributionMessage.serialize()
    return m