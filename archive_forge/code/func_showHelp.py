import sys
from PyQt5.QtCore import PYQT_VERSION_STR, QDir, QFile
from .pyrcc import *
def showHelp(error):
    sys.stderr.write('PyQt5 resource compiler\n')
    if error:
        sys.stderr.write('pyrcc5: %s\n' % error)
    sys.stderr.write('Usage: pyrcc5 [options] <inputs>\n\nOptions:\n    -o file           Write output to file rather than stdout\n    -threshold level  Threshold to consider compressing files\n    -compress level   Compress input files by level\n    -root path        Prefix resource access path with root path\n    -no-compress      Disable all compression\n    -version          Display version\n    -help             Display this information\n')