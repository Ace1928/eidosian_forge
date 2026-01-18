import numpy as np
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import Emitter, WidgetParameterItem
def limitsChanged(self, param, limits):
    self.optsChanged(param, dict(limits=limits))