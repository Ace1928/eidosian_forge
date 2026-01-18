from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem
def setParameters(self, param, showTop=True):
    """
        Set the top-level :class:`Parameter <pyqtgraph.parametertree.Parameter>`
        to be displayed in this ParameterTree.

        If *showTop* is False, then the top-level parameter is hidden and only 
        its children will be visible. This is a convenience method equivalent 
        to::
        
            tree.clear()
            tree.addParameters(param, showTop)
        """
    self.clear()
    self.addParameters(param, showTop=showTop)