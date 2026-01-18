from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
@barHeight.setter
def barHeight(self, value):
    n = self.tops['F'][0] - self.bottoms['F'][0]
    x = self.tops['F'][1] - self.bottoms['F'][1]
    value = self.__dict__['_barHeight'] = 72 * min(max(value / 72.0, n), x)
    self.heightSize = (value - n) / (x - n)