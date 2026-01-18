import os
from twisted.spread import pb
def getFolderMessage(self, folder, name):
    if '/' in name:
        raise OSError("can only open files in '%s' directory'" % folder)
    with open(os.path.join(self.directory, 'new', name)) as fp:
        return fp.read()