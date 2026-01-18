from __future__ import annotations
from ..runtime import driver
def getGlobalStrides(self, args):
    t_globalDims = [int(e) for e in self.getGlobalDims(args)]
    t_globalStridesArgIdx = self.globalStridesArgIdx.copy()
    strides_in_elements = []
    for i in range(self.tensorRank):
        t = 1
        if t_globalStridesArgIdx[i] == -1:
            for ii in range(i):
                t *= t_globalDims[ii]
        elif t_globalStridesArgIdx[i] < 0:
            t = -1 - t_globalStridesArgIdx[i]
        else:
            new_idx = self.getOriginArgIdx(t_globalStridesArgIdx[i], args)
            t = args[new_idx]
        strides_in_elements.append(t)
    strides_in_elements = strides_in_elements[1:]
    strides_in_bytes = [e * self.bytes_from_type(self.tensorDataType) for e in strides_in_elements]
    return strides_in_bytes