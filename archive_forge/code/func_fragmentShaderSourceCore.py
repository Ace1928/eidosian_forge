import sys
import math
import numpy
import ctypes
from PySide2.QtCore import QCoreApplication, Signal, SIGNAL, SLOT, Qt, QSize, QPoint
from PySide2.QtGui import (QVector3D, QOpenGLFunctions, QOpenGLVertexArrayObject, QOpenGLBuffer,
from PySide2.QtWidgets import (QApplication, QWidget, QMessageBox, QHBoxLayout, QSlider,
from shiboken2 import VoidPtr
def fragmentShaderSourceCore(self):
    return '#version 150\n                in highp vec3 vert;\n                in highp vec3 vertNormal;\n                out highp vec4 fragColor;\n                uniform highp vec3 lightPos;\n                void main() {\n                   highp vec3 L = normalize(lightPos - vert);\n                   highp float NL = max(dot(normalize(vertNormal), L), 0.0);\n                   highp vec3 color = vec3(0.39, 1.0, 0.0);\n                   highp vec3 col = clamp(color * 0.2 + color * 0.8 * NL, 0.0, 1.0);\n                   fragColor = vec4(col, 1.0);\n                }'