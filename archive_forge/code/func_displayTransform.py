import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def displayTransform(self, transform):
    """Updates transform numbers in the ctrl widget."""
    tr = transform.saveState()
    self.transformGui.translateLabel.setText('Translate: (%f, %f)' % (tr['pos'][0], tr['pos'][1]))
    self.transformGui.rotateLabel.setText('Rotate: %f degrees' % tr['angle'])
    self.transformGui.scaleLabel.setText('Scale: (%f, %f)' % (tr['scale'][0], tr['scale'][1]))