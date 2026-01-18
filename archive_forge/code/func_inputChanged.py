import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def inputChanged(self, term, process=True):
    """Called whenever there is a change to the input value to this terminal.
        It may often be useful to override this function."""
    if self.isMultiValue():
        self.setValue({term: term.value(self)}, process=process)
    else:
        self.setValue(term.value(self), process=process)