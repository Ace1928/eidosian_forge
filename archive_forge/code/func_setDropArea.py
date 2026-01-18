from ..Qt import QtCore, QtGui, QtWidgets
def setDropArea(self, area):
    self.dropArea = area
    if area is None:
        self.hide()
    else:
        prgn = self.parent().rect()
        rgn = QtCore.QRect(prgn)
        w = min(30, int(prgn.width() / 3))
        h = min(30, int(prgn.height() / 3))
        if self.dropArea == 'left':
            rgn.setWidth(w)
        elif self.dropArea == 'right':
            rgn.setLeft(rgn.left() + prgn.width() - w)
        elif self.dropArea == 'top':
            rgn.setHeight(h)
        elif self.dropArea == 'bottom':
            rgn.setTop(rgn.top() + prgn.height() - h)
        elif self.dropArea == 'center':
            rgn.adjust(w, h, -w, -h)
        self.setGeometry(rgn)
        self.show()
    self.update()