import os
import pickle
from twisted.internet.address import UNIXAddress
from twisted.mail import smtp
from twisted.python import log
def loadMessages(self, messagePaths):
    self.messages = []
    self.names = []
    for message in messagePaths:
        with open(message + '-H', 'rb') as fp:
            messageContents = pickle.load(fp)
        fp = open(message + '-D')
        messageContents.append(fp)
        self.messages.append(messageContents)
        self.names.append(message)