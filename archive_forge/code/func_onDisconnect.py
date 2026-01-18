import time
import logging
from threading import Thread, Lock
from yowsup.layers import YowProtocolLayer, YowLayerEvent, EventCallback
from yowsup.common import YowConstants
from yowsup.layers.network import YowNetworkLayer
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from .protocolentities import *
@EventCallback(YowNetworkLayer.EVENT_STATE_DISCONNECT)
def onDisconnect(self, event):
    self.stop_thread()