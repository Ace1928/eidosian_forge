from ._constants import *
def checkgroup(self, gid):
    return gid < self.groups and self.groupwidths[gid] is not None