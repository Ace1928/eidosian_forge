import gc
import importlib
import weakref
import warnings
from ..graphicsItems.GridItem import GridItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
from . import CanvasTemplate_generic as ui_template
from .CanvasItem import CanvasItem, GroupCanvasItem
from .CanvasManager import CanvasManager
def itemVisibilityChanged(self, item):
    listItem = item.listItem
    checked = listItem.checkState(0) == QtCore.Qt.CheckState.Checked
    vis = item.isVisible()
    if vis != checked:
        if vis:
            listItem.setCheckState(0, QtCore.Qt.CheckState.Checked)
        else:
            listItem.setCheckState(0, QtCore.Qt.CheckState.Unchecked)