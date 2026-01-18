import time
import logging
from threading import Thread, Lock
from yowsup.layers import YowProtocolLayer, YowLayerEvent, EventCallback
from yowsup.common import YowConstants
from yowsup.layers.network import YowNetworkLayer
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from .protocolentities import *
def stop_thread(self):
    if self._pingThread:
        self.__logger.debug('stopping ping thread')
        if self._pingThread:
            self._pingThread.stop()
            self._pingThread = None
        self._pingQueue = {}