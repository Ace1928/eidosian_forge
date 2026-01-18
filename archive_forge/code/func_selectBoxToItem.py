import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def selectBoxToItem(self):
    """Move/scale the selection box so it fits the item's bounding rect. (assumes item is not rotated)"""
    self.itemRect = self._graphicsItem.boundingRect()
    rect = self._graphicsItem.mapRectToParent(self.itemRect)
    self.selectBox.blockSignals(True)
    self.selectBox.setPos([rect.x(), rect.y()])
    self.selectBox.setSize(rect.size())
    self.selectBox.setAngle(0)
    self.selectBoxBase = self.selectBox.getState().copy()
    self.selectBox.blockSignals(False)