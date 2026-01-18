from __future__ import unicode_literals
from PySide2 import QtCore, QtGui, QtWidgets
import classwizard_rc
def initializePage(self):
    finishText = self.wizard().buttonText(QtWidgets.QWizard.FinishButton)
    finishText.replace('&', '')
    self.label.setText('Click %s to generate the class skeleton.' % finishText)