import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def setZValue(self, z):
    self.opts['z'] = z
    if z is not None:
        self._graphicsItem.setZValue(z)