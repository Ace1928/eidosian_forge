import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def apoptose(self, propagate=True):
    if self.topContainer is None or self.topContainer.count() == 0:
        self.topContainer = None
        if self.temporary and self.home is not None:
            self.home.removeTempArea(self)