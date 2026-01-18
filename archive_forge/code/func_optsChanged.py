import numpy as np
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import Emitter, WidgetParameterItem
def optsChanged(self, param, opts):
    try:
        super().optsChanged(param, opts)
    except AttributeError:
        pass
    span = opts.get('span', None)
    if span is None:
        step = opts.get('step', 1)
        start, stop = opts.get('limits', param.opts['limits'])
        span = np.arange(start, stop + step, step)
    precision = opts.get('precision', 2)
    if precision is not None:
        span = span.round(precision)
    self.span = span
    self.charSpan = np.char.array(span)
    w = self.slider
    w.setMinimum(0)
    w.setMaximum(len(span) - 1)
    if 'suffix' in opts:
        self.setSuffix(opts['suffix'])