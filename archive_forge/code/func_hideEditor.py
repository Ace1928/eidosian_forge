import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def hideEditor(self):
    self.widget.hide()
    self.displayLabel.show()