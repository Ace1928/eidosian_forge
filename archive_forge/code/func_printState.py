import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def printState(self, state=None, name='Main'):
    if state is None:
        state = self.saveState()
    print('=== %s dock area ===' % name)
    if state['main'] is None:
        print('   (empty)')
    else:
        self._printAreaState(state['main'])
    for i, float in enumerate(state['float']):
        self.printState(float[0], name='float %d' % i)