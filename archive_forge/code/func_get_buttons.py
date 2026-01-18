import jupyter_rfb
import numpy as np
from .. import functions as fn
from .. import graphicsItems, widgets
from ..Qt import QtCore, QtGui
def get_buttons(evt_buttons):
    NoButton = QtCore.Qt.MouseButton.NoButton
    btns = NoButton
    for x in evt_buttons:
        btns |= MBLUT.get(x, NoButton)
    return btns