import re
import itertools
def setupTypeNumber(self, test):
    bits = QRUtil.getBCHTypeNumber(self.version)
    for i in range(18):
        mod = not test and bits >> i & 1 == 1
        self.modules[i // 3][i % 3 + self.moduleCount - 8 - 3] = mod
    for i in range(18):
        mod = not test and bits >> i & 1 == 1
        self.modules[i % 3 + self.moduleCount - 8 - 3][i // 3] = mod