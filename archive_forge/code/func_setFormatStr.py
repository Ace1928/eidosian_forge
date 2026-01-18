from time import perf_counter
from .. import functions as fn
from ..Qt import QtWidgets
def setFormatStr(self, text):
    self.formatStr = text
    self.update()