from ._constants import *
def closegroup(self, gid, p):
    self.groupwidths[gid] = p.getwidth()