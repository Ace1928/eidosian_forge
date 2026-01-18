import inspect
import weakref
from .Qt import QtCore, QtWidgets
def checkForChildren(self, obj):
    """Return true if we should automatically search the children of this object for more."""
    iface = self.interface(obj)
    return len(iface) > 3 and iface[3]