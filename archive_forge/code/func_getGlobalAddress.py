from __future__ import annotations
from ..runtime import driver
def getGlobalAddress(self, args):
    idx = self.getOriginArgIdx(self.globalAddressArgIdx, args)
    return args[idx]