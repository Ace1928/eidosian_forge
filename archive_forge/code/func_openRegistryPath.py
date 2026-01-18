import sys
from PySide2 import QtCore, QtGui, QtWidgets
def openRegistryPath(self):
    path, ok = QtWidgets.QInputDialog.getText(self, 'Open Registry Path', 'Enter the path in the Windows registry:', QtWidgets.QLineEdit.Normal, 'HKEY_CURRENT_USER\\')
    if ok and path != '':
        settings = QtCore.QSettings(path, QtCore.QSettings.NativeFormat)
        self.setSettingsObject(settings)
        self.fallbacksAct.setEnabled(False)