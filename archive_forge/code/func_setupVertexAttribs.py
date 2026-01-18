from argparse import ArgumentParser, RawTextHelpFormatter
import numpy
import sys
from textwrap import dedent
from PySide2.QtCore import QCoreApplication, QLibraryInfo, QSize, QTimer, Qt
from PySide2.QtGui import (QMatrix4x4, QOpenGLBuffer, QOpenGLContext, QOpenGLShader,
from PySide2.QtWidgets import (QApplication, QHBoxLayout, QMessageBox, QPlainTextEdit,
from PySide2.support import VoidPtr
def setupVertexAttribs(self):
    self.vbo.bind()
    self.program.setAttributeBuffer(self.posAttr, GL.GL_FLOAT, 0, 2)
    self.program.setAttributeBuffer(self.colAttr, GL.GL_FLOAT, 4 * vertices.size, 3)
    self.program.enableAttributeArray(self.posAttr)
    self.program.enableAttributeArray(self.colAttr)
    self.vbo.release()