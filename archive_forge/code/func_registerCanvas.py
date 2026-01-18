from ..Qt import QtCore, QtWidgets
import weakref
def registerCanvas(self, canvas, name):
    n2 = name
    i = 0
    while n2 in self.canvases:
        n2 = '%s_%03d' % (name, i)
        i += 1
    self.canvases[n2] = canvas
    self.sigCanvasListChanged.emit()
    return n2