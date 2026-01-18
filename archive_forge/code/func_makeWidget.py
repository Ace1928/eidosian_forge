from ... import functions as fn
from ...widgets.ColorButton import ColorButton
from .basetypes import SimpleParameter, WidgetParameterItem
def makeWidget(self):
    w = ColorButton()
    w.sigChanged = w.sigColorChanged
    w.sigChanging = w.sigColorChanging
    w.value = w.color
    w.setValue = w.setColor
    self.hideWidget = False
    w.setFlat(True)
    return w