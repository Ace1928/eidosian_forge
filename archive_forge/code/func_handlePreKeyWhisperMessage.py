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
def handlePreKeyWhisperMessage(self, node):
    pkMessageProtocolEntity = EncryptedMessageProtocolEntity.fromProtocolTreeNode(node)
    enc = pkMessageProtocolEntity.getEnc(EncProtocolEntity.TYPE_PKMSG)
    plaintext = self.manager.decrypt_pkmsg(pkMessageProtocolEntity.getAuthor(False), enc.getData(), enc.getVersion() == 2)
    if enc.getVersion() == 2:
        self.parseAndHandleMessageProto(pkMessageProtocolEntity, plaintext)
    node = pkMessageProtocolEntity.toProtocolTreeNode()
    node.addChild(ProtoProtocolEntity(plaintext, enc.getMediaType()).toProtocolTreeNode())
    self.toUpper(node)