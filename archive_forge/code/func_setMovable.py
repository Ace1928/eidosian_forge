import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def setMovable(self, m):
    self.opts['movable'] = m
    if m:
        self.resetTransformBtn.show()
        self.copyBtn.show()
        self.pasteBtn.show()
    else:
        self.resetTransformBtn.hide()
        self.copyBtn.hide()
        self.pasteBtn.hide()