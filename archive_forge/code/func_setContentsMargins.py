from .. import functions as fn
from ..Qt import QtWidgets
from .GraphicsWidget import GraphicsWidget
from .LabelItem import LabelItem
from .PlotItem import PlotItem
from .ViewBox import ViewBox
def setContentsMargins(self, *args):
    self.layout.setContentsMargins(*args)