import sys
from PyQt5.QtCore import PYQT_VERSION_STR, QDir, QFile
from .pyrcc import *
def processResourceFile(filenamesIn, filenameOut, listFiles):
    if verbose:
        sys.stderr.write('PyQt5 resource compiler\n')
    library = RCCResourceLibrary()
    library.setInputFiles(filenamesIn)
    library.setVerbose(verbose)
    library.setCompressLevel(compressLevel)
    library.setCompressThreshold(compressThreshold)
    library.setResourceRoot(resourceRoot)
    if not library.readFiles():
        return False
    if filenameOut == '-':
        filenameOut = ''
    if listFiles:
        if filenameOut:
            try:
                out_fd = open(filenameOut, 'w')
            except Exception:
                sys.stderr.write('Unable to open %s for writing\n' % filenameOut)
                return False
        else:
            out_fd = sys.stdout
        for df in library.dataFiles():
            out_fd.write('%s\n' % QDir.cleanPath(df))
        if out_fd is not sys.stdout:
            out_fd.close()
        return True
    return library.output(filenameOut)