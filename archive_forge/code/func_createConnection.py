from yowsup.layers import YowLayer, YowLayerEvent, EventCallback
from yowsup.layers.network.layer_interface import YowNetworkLayerInterface
from yowsup.layers.network.dispatcher.dispatcher import ConnectionCallbacks
from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
from yowsup.layers.network.dispatcher.dispatcher_socket import SocketConnectionDispatcher
from yowsup.layers.network.dispatcher.dispatcher_asyncore import AsyncoreConnectionDispatcher
import logging
def createConnection(self):
    self._disconnect_reason = None
    self._dispatcher = self.__create_dispatcher(self.getProp(self.PROP_DISPATCHER, self.DISPATCHER_DEFAULT))
    self.state = self.__class__.STATE_CONNECTING
    endpoint = self.getProp(self.__class__.PROP_ENDPOINT)
    logger.info('Connecting to %s:%s' % endpoint)
    self._dispatcher.connect(endpoint)