from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtOpenGL():
    type_map.update({'GLbitfield': int, 'GLenum': int, 'GLfloat': float, 'GLint': int, 'GLuint': int, 'PySide2.QtOpenGL.GLint': int, 'PySide2.QtOpenGL.GLuint': int})
    return locals()