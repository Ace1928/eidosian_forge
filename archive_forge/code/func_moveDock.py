import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def moveDock(self, dock, position, neighbor):
    """
        Move an existing Dock to a new location. 
        """
    if position in ['left', 'right', 'top', 'bottom'] and neighbor is not None and (neighbor.container() is not None) and (neighbor.container().type() == 'tab'):
        neighbor = neighbor.container()
    self.addDock(dock, position, neighbor)