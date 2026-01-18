import time
import logging
from threading import Thread, Lock
from yowsup.layers import YowProtocolLayer, YowLayerEvent, EventCallback
from yowsup.common import YowConstants
from yowsup.layers.network import YowNetworkLayer
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from .protocolentities import *
def gotPong(self, pingId):
    self._pingQueueLock.acquire()
    if pingId in self._pingQueue:
        self._pingQueue = {}
    self._pingQueueLock.release()