from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def childrenValue(self):
    vals = [self.forward[p.name()] for p in self.children() if p.value()]
    exclusive = self.opts['exclusive']
    if not vals and exclusive:
        return None
    elif exclusive:
        return vals[0]
    else:
        return vals