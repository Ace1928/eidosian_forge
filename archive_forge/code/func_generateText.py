from time import perf_counter
from .. import functions as fn
from ..Qt import QtWidgets
def generateText(self):
    if len(self.values) == 0:
        return ''
    avg = self.averageValue()
    val = self.values[-1][1]
    if self.siPrefix:
        return fn.siFormat(avg, suffix=self.suffix)
    else:
        return self.formatStr.format(value=val, avgValue=avg, suffix=self.suffix)