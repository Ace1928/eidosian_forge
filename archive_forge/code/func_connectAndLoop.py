from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
import socket
import logging
def connectAndLoop(self, host):
    socket = self.socket
    self.connectionCallbacks.onConnecting()
    try:
        socket.connect(host)
        self.connectionCallbacks.onConnected()
        while True:
            data = socket.recv(1024)
            if len(data):
                self.connectionCallbacks.onRecvData(data)
            else:
                break
        self.connectionCallbacks.onDisconnected()
    except Exception as e:
        logger.error(e)
        self.connectionCallbacks.onConnectionError(e)
    finally:
        self.socket = None
        socket.close()