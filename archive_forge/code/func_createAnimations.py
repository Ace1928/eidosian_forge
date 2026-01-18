from PySide2 import QtCore, QtGui, QtWidgets
import appchooser_rc
def createAnimations(objects, machine):
    for obj in objects:
        animation = QtCore.QPropertyAnimation(obj, b'geometry', obj)
        machine.addDefaultAnimation(animation)