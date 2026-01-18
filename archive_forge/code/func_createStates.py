from PySide2 import QtCore, QtGui, QtWidgets
import appchooser_rc
def createStates(objects, selectedRect, parent):
    for obj in objects:
        state = QtCore.QState(parent)
        state.assignProperty(obj, 'geometry', selectedRect)
        parent.addTransition(obj.clicked, state)