import numpy as np
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import Emitter, WidgetParameterItem
def prettyTextValue(self, v):
    if self._suffix is None:
        suffixTxt = ''
    else:
        suffixTxt = f' {self._suffix}'
    format_ = self.param.opts.get('format', None)
    cspan = self.charSpan
    if format_ is None:
        format_ = f'{{0:>{cspan.dtype.itemsize}}}{suffixTxt}'
    return format_.format(cspan[v].decode())