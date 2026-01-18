import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def restoreTransform(self, tr):
    try:
        self.userTransform = SRTTransform(tr)
        self.updateTransform()
        self.selectBoxFromUser()
        self.sigTransformChanged.emit(self)
        self.sigTransformChangeFinished.emit(self)
    except:
        self.userTransform = SRTTransform()
        debug.printExc('Failed to load transform:')