from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def maybeSigChanged(self, val):
    """
        Make sure to only activate on a "true" value, since an exclusive button group fires once to deactivate
        the old option and once to activate the new selection
        """
    if not val:
        return
    self.emitter.sigChanged.emit(self, val)