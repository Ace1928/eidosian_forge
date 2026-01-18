from .. import exporters as exporters
from .. import functions as fn
from ..graphicsItems.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtWidgets
from . import exportDialogTemplate_generic as ui_template
def exportFormatChanged(self, item, prev):
    if item is None:
        self.currentExporter = None
        self.ui.paramTree.clear()
        return
    expClass = item.expClass
    exp = expClass(item=self.ui.itemTree.currentItem().gitem)
    params = exp.parameters()
    if params is None:
        self.ui.paramTree.clear()
    else:
        self.ui.paramTree.setParameters(params)
    self.currentExporter = exp
    self.ui.copyBtn.setEnabled(exp.allowCopy)