from argparse import ArgumentParser, RawTextHelpFormatter
import numpy
import sys
from textwrap import dedent
from PySide2.QtCore import QCoreApplication, QLibraryInfo, QSize, QTimer, Qt
from PySide2.QtGui import (QMatrix4x4, QOpenGLBuffer, QOpenGLContext, QOpenGLShader,
from PySide2.QtWidgets import (QApplication, QHBoxLayout, QMessageBox, QPlainTextEdit,
from PySide2.support import VoidPtr
def slotTimer(self):
    self.render()
    self.angle += 1