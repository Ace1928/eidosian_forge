import sys
from collections import OrderedDict
from ..Qt import QtWidgets
def indexChanged(self, index):
    if self._ignoreIndexChange:
        return
    self._chosenText = self.currentText()