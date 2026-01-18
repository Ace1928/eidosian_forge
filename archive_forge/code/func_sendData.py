from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
import asyncore
import logging
import socket
import traceback
def sendData(self, data):
    if self._connected:
        self.out_buffer = self.out_buffer + data
        self.initiate_send()
    else:
        logger.warn('Attempted to send %d bytes while still not connected' % len(data))