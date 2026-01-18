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
@EventCallback(YowAuthenticationProtocolLayer.EVENT_AUTHED)
def onAuthed(self, yowLayerEvent):
    if yowLayerEvent.getArg('passive') and len(self._unsent_prekeys):
        logger.debug('SHOULD FLUSH KEYS %d NOW!!' % len(self._unsent_prekeys))
        self.flush_keys(self.manager.load_latest_signed_prekey(generate=True), self._unsent_prekeys[:], reboot_connection=True)
        self._unsent_prekeys = []