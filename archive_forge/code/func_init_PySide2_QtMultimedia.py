from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtMultimedia():
    import PySide2.QtMultimediaWidgets
    check_module(PySide2.QtMultimediaWidgets)
    type_map.update({'QGraphicsVideoItem': PySide2.QtMultimediaWidgets.QGraphicsVideoItem, 'QVideoWidget': PySide2.QtMultimediaWidgets.QVideoWidget})
    return locals()