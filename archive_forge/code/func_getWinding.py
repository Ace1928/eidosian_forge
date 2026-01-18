from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import solveQuadratic, solveCubic
def getWinding(self):
    if self.firstPoint is not None:
        self.closePath()
    return self.intersectionCount