from ..Qt import QtCore, QtWidgets
def setToolTip(self, text=None, temporary=False):
    if text is None:
        text = self.origTip
    QtWidgets.QPushButton.setToolTip(self, text)
    if not temporary:
        self.origTip = text