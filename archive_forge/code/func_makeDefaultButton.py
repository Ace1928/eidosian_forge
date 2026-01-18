import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def makeDefaultButton(self):
    defaultBtn = QtWidgets.QPushButton()
    defaultBtn.setAutoDefault(False)
    defaultBtn.setFixedWidth(20)
    defaultBtn.setFixedHeight(20)
    defaultBtn.setIcon(icons.getGraphIcon('default'))
    defaultBtn.clicked.connect(self.defaultClicked)
    return defaultBtn