from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem
def focusPrevious(self, item):
    self.focusNext(item, forward=False)