import os
from twisted.spread import pb
def getSubFolder(self, name):
    if '/' in name or name[0] == '.':
        raise OSError('invalid name')
    return Maildir('.', os.path.join(self.getRoot(), name))