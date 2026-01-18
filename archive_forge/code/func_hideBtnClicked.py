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
def hideBtnClicked(self):
    ctrlSize = self.ui.splitter.sizes()[1]
    if ctrlSize == 0:
        cs = self.ctrlSize
        w = self.ui.splitter.size().width()
        if cs > w:
            cs = w - 20
        self.ui.splitter.setSizes([w - cs, cs])
        self.hideBtn.setText('>')
    else:
        self.ctrlSize = ctrlSize
        self.ui.splitter.setSizes([100, 0])
        self.hideBtn.setText('<')
    self.resizeEvent()