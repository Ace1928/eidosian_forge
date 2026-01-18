import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def mirrorY(self):
    if not self.isMovable():
        return
    inv = SRTTransform()
    inv.scale(-1, 1)
    self.userTransform = self.userTransform * inv
    self.updateTransform()
    self.selectBoxFromUser()
    self.sigTransformChangeFinished.emit(self)