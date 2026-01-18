from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtWidgets():
    from PySide2.QtWidgets import QWidget, QMessageBox, QStyleOption, QStyleHintReturn, QStyleOptionComplex
    from PySide2.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem
    type_map.update({'QMessageBox.StandardButtons(Yes | No)': Instance('QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.No)'), 'QVector< int >()': [], 'QWidget.RenderFlags(DrawWindowBackground | DrawChildren)': Instance('QWidget.RenderFlags(QWidget.DrawWindowBackground | QWidget.DrawChildren)'), 'SH_Default': QStyleHintReturn.SH_Default, 'SO_Complex': QStyleOptionComplex.SO_Complex, 'SO_Default': QStyleOption.SO_Default, 'static_cast<Qt.MatchFlags>(Qt.MatchExactly|Qt.MatchCaseSensitive)': Instance('Qt.MatchFlags(Qt.MatchExactly | Qt.MatchCaseSensitive)'), 'Type': PySide2.QtWidgets.QListWidgetItem.Type})
    return locals()