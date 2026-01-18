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
def adjustId(self, _id):
    _id = format(_id, 'x')
    zfiller = len(_id) if len(_id) % 2 == 0 else len(_id) + 1
    _id = _id.zfill(zfiller if zfiller > 6 else 6)
    return binascii.unhexlify(_id)