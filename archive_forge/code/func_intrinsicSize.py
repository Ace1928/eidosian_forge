from PySide2 import QtCore, QtGui, QtWidgets, QtSvg
def intrinsicSize(self, doc, posInDocument, format):
    renderer = QtSvg.QSvgRenderer(format.property(Window.SvgData).toByteArray())
    size = renderer.defaultSize()
    if size.height() > 25:
        size *= 25.0 / size.height()
    return QtCore.QSizeF(size)