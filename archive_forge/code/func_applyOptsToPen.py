import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
def applyOptsToPen(self, **opts):
    paramNames = set(opts).intersection(self.names)
    with self.treeChangeBlocker():
        if 'value' in opts:
            pen = self.mkPen(opts.pop('value'))
            if not fn.eq(pen, self.pen):
                self.updateFromPen(self, pen)
        penOpts = {}
        for kk in paramNames:
            penOpts[kk] = opts[kk]
            self[kk] = opts[kk]
    return penOpts