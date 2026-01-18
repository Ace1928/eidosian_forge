from .layer_base import AxolotlBaseLayer
from yowsup.layers import YowLayerEvent, EventCallback
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers.axolotl.protocolentities import *
from yowsup.layers.auth.layer_authentication import YowAuthenticationProtocolLayer
from yowsup.layers.protocol_acks.protocolentities import OutgoingAckProtocolEntity
from axolotl.util.hexutil import HexUtil
from axolotl.ecc.curve import Curve
import logging
import binascii
def onIdentityChangeEncryptNotification(self, protocoltreenode):
    entity = IdentityChangeEncryptNotification.fromProtocolTreeNode(protocoltreenode)
    ack = OutgoingAckProtocolEntity(protocoltreenode['id'], 'notification', protocoltreenode['type'], protocoltreenode['from'])
    self.toLower(ack.toProtocolTreeNode())
    self.getKeysFor([entity.getFrom(True)], resultClbk=lambda _, __: None, reason='identity')