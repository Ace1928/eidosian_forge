from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
import asyncore
import logging
import socket
import traceback
def handle_read(self):
    data = self.recv(1024)
    self.connectionCallbacks.onRecvData(data)