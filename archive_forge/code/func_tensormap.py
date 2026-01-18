from __future__ import annotations
from ..runtime import driver
def tensormap(self, args):
    return driver.utils.cuTensorMapEncodeTiled(self.getTensorMapDataType(), self.getTensorRank(), self.getGlobalAddress(args), self.getGlobalDims(args), self.getGlobalStrides(args), self.getBoxDims(), self.getElementStrides(), self.getInterleave(), self.getSwizzle(), self.getL2Promotion(), self.getOobFill())