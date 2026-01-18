import numpy as np
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import Emitter, WidgetParameterItem
def spanToSliderValue(self, v):
    return int(np.argmin(np.abs(self.span - v)))