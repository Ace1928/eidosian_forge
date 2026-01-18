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
def multiSelectBoxMoved(self):
    transform = self.multiSelectBox.getGlobalTransform()
    for ci in self.selectedItems():
        ci.setTemporaryTransform(transform)
        ci.sigTransformChanged.emit(ci)