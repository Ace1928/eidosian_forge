from yowsup.common import YowConstants
from yowsup.layers import YowLayerEvent, YowProtocolLayer, EventCallback
from yowsup.layers.network import YowNetworkLayer
from .protocolentities import *
from .layer_interface_authentication import YowAuthenticationProtocolLayerInterface
from .protocolentities import StreamErrorProtocolEntity
import logging
def handleSuccess(self, node):
    successEvent = YowLayerEvent(self.__class__.EVENT_AUTHED, passive=self.getProp(self.__class__.PROP_PASSIVE))
    self.broadcastEvent(successEvent)
    nodeEntity = SuccessProtocolEntity.fromProtocolTreeNode(node)
    self.toUpper(nodeEntity)