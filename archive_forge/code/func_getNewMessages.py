import os
from twisted.spread import pb
def getNewMessages(self):
    return os.listdir(os.path.join(self.directory, 'new'))