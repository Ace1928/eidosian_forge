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
def sendToContact(self, node):
    recipient_id = node['to'].split('@')[0]
    protoNode = node.getChild('proto')
    messageData = protoNode.getData()
    ciphertext = self.manager.encrypt(recipient_id, messageData)
    mediaType = protoNode['mediatype']
    return self.sendEncEntities(node, [EncProtocolEntity(EncProtocolEntity.TYPE_MSG if ciphertext.__class__ == WhisperMessage else EncProtocolEntity.TYPE_PKMSG, 2, ciphertext.serialize(), mediaType)])