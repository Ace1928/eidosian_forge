import os
from twisted.spread import pb
def getSubFolders(self):
    return os.listdir(self.getRoot())