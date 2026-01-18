import jupyter_rfb
import numpy as np
from .. import functions as fn
from .. import graphicsItems, widgets
from ..Qt import QtCore, QtGui
def get_modifiers(evt_modifiers):
    NoModifier = QtCore.Qt.KeyboardModifier.NoModifier
    mods = NoModifier
    for x in evt_modifiers:
        mods |= KMLUT.get(x, NoModifier)
    return mods