from PySide2 import QtCore, QtGui, QtWidgets, QtPrintSupport
def printFile(self):
    editor = self.letters.currentWidget()
    printer = QtPrintSupport.QPrinter()
    dialog = QtPrintSupport.QPrintDialog(printer, self)
    dialog.setWindowTitle('Print Document')
    if editor.textCursor().hasSelection():
        dialog.addEnabledOption(QtPrintSupport.QAbstractPrintDialog.PrintSelection)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return
    editor.print_(printer)