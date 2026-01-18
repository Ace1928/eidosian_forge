import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
def penPropertySetter(self, p, value):
    boundPen = self.pen
    setName = f'set{cap_first(p.name())}'
    getattr(boundPen.__class__, setName)(boundPen, value)
    self.sigValueChanging.emit(self, boundPen)