from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem
def itemChangedEvent(self, item, col):
    if hasattr(item, 'columnChangedEvent'):
        item.columnChangedEvent(col)