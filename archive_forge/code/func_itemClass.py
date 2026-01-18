from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
@property
def itemClass(self):
    if self.opts.get('type') == 'bool':
        return BoolParameterItem
    else:
        return RadioParameterItem