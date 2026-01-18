import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def removeTempArea(self, area):
    self.tempAreas.remove(area)
    area.window().close()