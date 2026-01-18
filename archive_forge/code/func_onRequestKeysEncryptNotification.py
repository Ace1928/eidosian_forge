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
def onRequestKeysEncryptNotification(self, protocolTreeNode):
    entity = RequestKeysEncryptNotification.fromProtocolTreeNode(protocolTreeNode)
    ack = OutgoingAckProtocolEntity(protocolTreeNode['id'], 'notification', protocolTreeNode['type'], protocolTreeNode['from'])
    self.toLower(ack.toProtocolTreeNode())
    self.flush_keys(self.manager.generate_signed_prekey(), self.manager.level_prekeys(force=True))