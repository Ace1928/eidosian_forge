import os
import re
from ..GraphicsScene import GraphicsScene
from ..Qt import QtCore, QtWidgets
from ..widgets.FileDialog import FileDialog
def fileSaveDialog(self, filter=None, opts=None):
    if opts is None:
        opts = {}
    self.fileDialog = FileDialog()
    self.fileDialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
    self.fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
    if filter is not None:
        if isinstance(filter, str):
            self.fileDialog.setNameFilter(filter)
        elif isinstance(filter, list):
            self.fileDialog.setNameFilters(filter)
    global LastExportDirectory
    exportDir = LastExportDirectory
    if exportDir is not None:
        self.fileDialog.setDirectory(exportDir)
    self.fileDialog.show()
    self.fileDialog.opts = opts
    self.fileDialog.fileSelected.connect(self.fileSaveFinished)
    return