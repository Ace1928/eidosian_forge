import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def applyTemporaryTransform(self):
    """Collapses tempTransform into UserTransform, resets tempTransform"""
    self.userTransform = self.userTransform * self.tempTransform
    self.resetTemporaryTransform()
    self.selectBoxFromUser()