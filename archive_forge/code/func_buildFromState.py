import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def buildFromState(self, state, docks, root, depth=0, missing='error'):
    typ, contents, state = state
    if typ == 'dock':
        try:
            obj = docks[contents]
            del docks[contents]
        except KeyError:
            if missing == 'error':
                raise Exception('Cannot restore dock state; no dock with name "%s"' % contents)
            elif missing == 'create':
                obj = Dock(name=contents)
            elif missing == 'ignore':
                return
            else:
                raise ValueError('"missing" argument must be one of "error", "create", or "ignore".')
    else:
        obj = self.makeContainer(typ)
    root.insert(obj, 'after')
    if typ != 'dock':
        for o in contents:
            self.buildFromState(o, docks, obj, depth + 1, missing=missing)
        obj.apoptose(propagate=False)
        obj.restoreState(state)