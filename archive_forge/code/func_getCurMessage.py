import os
from twisted.spread import pb
def getCurMessage(self, name):
    return self.getFolderMessage('cur', name)