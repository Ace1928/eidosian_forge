from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
from .ViewBox import ViewBox
def setViewList(self, views):
    names = ['']
    self.viewMap.clear()
    for v in views:
        name = v.name
        if name is None:
            continue
        names.append(name)
        self.viewMap[name] = v
    for i in [0, 1]:
        c = self.ctrl[i].linkCombo
        current = c.currentText()
        c.blockSignals(True)
        changed = True
        try:
            c.clear()
            for name in names:
                c.addItem(name)
                if name == current:
                    changed = False
                    c.setCurrentIndex(c.count() - 1)
        finally:
            c.blockSignals(False)
        if changed:
            c.setCurrentIndex(0)
            c.currentIndexChanged.emit(c.currentIndex())