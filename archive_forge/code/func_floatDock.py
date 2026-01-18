import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def floatDock(self, dock):
    """Removes *dock* from this DockArea and places it in a new window."""
    area = self.addTempArea()
    area.win.resize(dock.size())
    area.moveDock(dock, 'top', None)