from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import solveQuadratic, solveCubic
def setTestPoint(self, testPoint, evenOdd=False):
    """Set the point to test. Call this _before_ the outline gets drawn."""
    self.testPoint = testPoint
    self.evenOdd = evenOdd
    self.firstPoint = None
    self.intersectionCount = 0