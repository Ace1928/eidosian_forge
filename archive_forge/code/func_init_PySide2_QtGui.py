from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtGui():
    from PySide2.QtGui import QPageLayout, QPageSize
    type_map.update({'0.0f': 0.0, '1.0f': 1.0, 'GL_COLOR_BUFFER_BIT': GL_COLOR_BUFFER_BIT, 'GL_NEAREST': GL_NEAREST, 'int32_t': int, 'PySide2.QtCore.uint8_t': int, 'PySide2.QtGui.QGenericMatrix': Missing('PySide2.QtGui.QGenericMatrix'), 'PySide2.QtGui.QPlatformSurface': int, 'QList< QTouchEvent.TouchPoint >()': [], 'QPixmap()': Default('PySide2.QtGui.QPixmap'), 'QVector< QTextLayout.FormatRange >()': [], 'uint32_t': int, 'uint8_t': int, 'USHRT_MAX': ushort_max, 'WId': WId})
    return locals()