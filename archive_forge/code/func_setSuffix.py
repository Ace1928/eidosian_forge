import numpy as np
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import Emitter, WidgetParameterItem
def setSuffix(self, suffix):
    self._suffix = suffix
    if hasattr(self, 'displayLabel'):
        self.updateDisplayLabel(self.slider.value())