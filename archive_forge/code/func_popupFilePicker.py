import os
import re
from ...Qt import QtCore, QtGui, QtWidgets
from ..Parameter import Parameter
from .str import StrParameterItem
def popupFilePicker(parent=None, windowTitle='', nameFilter='', directory=None, selectFile=None, relativeTo=None, **kwargs):
    """
    Thin wrapper around Qt file picker dialog. Used internally so all options are consistent
    among all requests for external file information

    ============== ========================================================
    **Arguments:**
    parent         Dialog parent
    windowTitle    Title of dialog window
    nameFilter     File filter as required by the Qt dialog
    directory      Where in the file system to open this dialog
    selectFile     File to preselect
    relativeTo     Parent directory that, if provided, will be removed from the prefix of all returned paths. So,
                   if '/my/text/file.txt' was selected, and `relativeTo='/my/text/'`, the return value would be
                   'file.txt'. This uses os.path.relpath under the hood, so expect that behavior.
    kwargs         Any enum value accepted by a QFileDialog and its value. Values can be a string or list of strings,
                   i.e. fileMode='AnyFile', options=['ShowDirsOnly', 'DontResolveSymlinks'], acceptMode='AcceptSave'
    ============== ========================================================

    """
    fileDlg = QtWidgets.QFileDialog(parent)
    _set_filepicker_kwargs(fileDlg, **kwargs)
    fileDlg.setModal(True)
    if directory is not None:
        fileDlg.setDirectory(directory)
    fileDlg.setNameFilter(nameFilter)
    if selectFile is not None:
        fileDlg.selectFile(selectFile)
    fileDlg.setWindowTitle(windowTitle)
    if fileDlg.exec():
        singleExtReg = '(\\.\\w+)'
        suffMatch = re.search(f'({singleExtReg}+)', fileDlg.selectedNameFilter())
        if suffMatch:
            ext = suffMatch.group(1)
            if ext.startswith('.'):
                ext = ext[1:]
            fileDlg.setDefaultSuffix(ext)
        fList = fileDlg.selectedFiles()
    else:
        fList = []
    if relativeTo is not None:
        fList = [os.path.relpath(file, relativeTo) for file in fList]
    fList = [os.path.normpath(file) for file in fList]
    if fileDlg.fileMode() == fileDlg.FileMode.ExistingFiles:
        return fList
    elif len(fList) > 0:
        return fList[0]
    else:
        return None