import time
import logging
from threading import Thread, Lock
from yowsup.layers import YowProtocolLayer, YowLayerEvent, EventCallback
from yowsup.common import YowConstants
from yowsup.layers.network import YowNetworkLayer
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from .protocolentities import *
def onPong(self, protocolTreeNode, pingEntity):
    self.gotPong(pingEntity.getId())
    self.toUpper(ResultIqProtocolEntity.fromProtocolTreeNode(protocolTreeNode))