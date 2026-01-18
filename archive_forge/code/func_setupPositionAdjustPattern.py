import re
import itertools
def setupPositionAdjustPattern(self):
    pos = QRUtil.getPatternPosition(self.version)
    maxpos = self.moduleCount - 8
    for row, col in itertools.product(pos, pos):
        if col <= 8 and (row <= 8 or row >= maxpos):
            continue
        elif col >= maxpos and row <= 8:
            continue
        for r, data in enumerate(self._positionAdjustPattern):
            self.modules[row + r - 2][col - 2:col + 3] = data