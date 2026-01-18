import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
def setDefault(self, val, **kwargs):
    pen = self._interpretValue(val)
    with self.treeChangeBlocker():
        for opt in self.names:
            if isinstance(self[opt], bool):
                attrName = f'is{opt.title()}'
            else:
                attrName = opt
            self.child(opt).setDefault(getattr(pen, attrName)(), **kwargs)
        out = super().setDefault(val, **kwargs)
    return out