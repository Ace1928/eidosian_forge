from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def drawHumanReadable(self):
    if self.humanReadable:
        hcz = self.horizontalClearZone
        vcz = self.verticalClearZone
        fontName = self.fontName
        fontSize = self.fontSize
        y = self.barHeight + 2 * vcz + 0.2 * fontSize
        self.annotate(hcz, y, self.value, fontName, fontSize)