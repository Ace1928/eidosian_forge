import os
from twisted.spread import pb
def deleteNewMessage(self, name):
    return self.deleteFolderMessage('new', name)