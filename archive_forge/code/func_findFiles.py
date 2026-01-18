from PySide2 import QtCore, QtGui, QtWidgets
def findFiles(self, files, text):
    progressDialog = QtWidgets.QProgressDialog(self)
    progressDialog.setCancelButtonText('&Cancel')
    progressDialog.setRange(0, len(files))
    progressDialog.setWindowTitle('Find Files')
    foundFiles = []
    for i in range(len(files)):
        progressDialog.setValue(i)
        progressDialog.setLabelText('Searching file number %d of %d...' % (i, len(files)))
        QtCore.qApp.processEvents()
        if progressDialog.wasCanceled():
            break
        inFile = QtCore.QFile(self.currentDir.absoluteFilePath(files[i]))
        if inFile.open(QtCore.QIODevice.ReadOnly):
            stream = QtCore.QTextStream(inFile)
            while not stream.atEnd():
                if progressDialog.wasCanceled():
                    break
                line = stream.readLine()
                if text in line:
                    foundFiles.append(files[i])
                    break
    progressDialog.close()
    return foundFiles