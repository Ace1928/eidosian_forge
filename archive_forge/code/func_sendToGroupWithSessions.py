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
def sendToGroupWithSessions(self, node, jidsNeedSenderKey=None, retryCount=0):
    """
        For each jid in jidsNeedSenderKey will create a pkmsg enc node with the associated jid.
        If retryCount > 0 and we have only one jidsNeedSenderKey, this is a retry requested by a specific participant
        and this message is to be directed at specific at that participant indicated by jidsNeedSenderKey[0]. In this
        case the participant's jid would go in the parent's EncryptedMessage and not into the enc node.
        """
    logger.debug('sendToGroupWithSessions(node=[omitted], jidsNeedSenderKey=%s, retryCount=%d)' % (jidsNeedSenderKey, retryCount))
    jidsNeedSenderKey = jidsNeedSenderKey or []
    groupJid = node['to']
    protoNode = node.getChild('proto')
    encEntities = []
    participant = jidsNeedSenderKey[0] if len(jidsNeedSenderKey) == 1 and retryCount > 0 else None
    if len(jidsNeedSenderKey):
        senderKeyDistributionMessage = self.manager.group_create_skmsg(groupJid)
        for jid in jidsNeedSenderKey:
            message = self.serializeSenderKeyDistributionMessageToProtobuf(node['to'], senderKeyDistributionMessage)
            if retryCount > 0:
                message.MergeFromString(protoNode.getData())
            ciphertext = self.manager.encrypt(jid.split('@')[0], message.SerializeToString())
            encEntities.append(EncProtocolEntity(EncProtocolEntity.TYPE_MSG if ciphertext.__class__ == WhisperMessage else EncProtocolEntity.TYPE_PKMSG, 2, ciphertext.serialize(), protoNode['mediatype'], jid=None if participant else jid))
    if not retryCount:
        messageData = protoNode.getData()
        ciphertext = self.manager.group_encrypt(groupJid, messageData)
        mediaType = protoNode['mediatype']
        encEntities.append(EncProtocolEntity(EncProtocolEntity.TYPE_SKMSG, 2, ciphertext, mediaType))
    self.sendEncEntities(node, encEntities, participant)