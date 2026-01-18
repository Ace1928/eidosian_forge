from ..Qt import QtCore, QtWidgets
def setStyleSheet(self, style=None, temporary=False):
    if style is None:
        style = self.origStyle
    QtWidgets.QPushButton.setStyleSheet(self, style)
    if not temporary:
        self.origStyle = style