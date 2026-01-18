from ..Qt import QtCore, QtGui, QtWidgets
from .PathButton import PathButton
def setCollapsed(self, c):
    if c == self._collapsed:
        return
    if c is True:
        self.collapseBtn.setPath(self.closePath)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred, closing=True)
    elif c is False:
        self.collapseBtn.setPath(self.openPath)
        self.setSizePolicy(self._lastSizePolicy)
    else:
        raise TypeError('Invalid argument %r; must be bool.' % c)
    for ch in self.children():
        if isinstance(ch, QtWidgets.QWidget) and ch is not self.collapseBtn:
            ch.setVisible(not c)
    self._collapsed = c
    self.sigCollapseChanged.emit(c)