import weakref
from math import atan2, degrees
from ..functions import clip_scalar
from ..Qt import QtCore, QtWidgets
from . import ArrowItem
from .GraphicsObject import GraphicsObject
def makeAnimation(self, prop='position', start=0.0, end=1.0, duration=10000, loop=1):
    if not isinstance(prop, bytes):
        prop = prop.encode('latin-1')
    anim = QtCore.QPropertyAnimation(self, prop)
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setLoopCount(loop)
    return anim