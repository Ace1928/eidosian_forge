import warnings
import weakref
from time import perf_counter, perf_counter_ns
from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
def sendClickEvent(self, ev):
    if self.dragItem is not None and hasattr(self.dragItem, 'mouseClickEvent'):
        ev.currentItem = self.dragItem
        self.dragItem.mouseClickEvent(ev)
    else:
        if self.lastHoverEvent is not None:
            acceptedItem = self.lastHoverEvent.clickItems().get(ev.button(), None)
        else:
            acceptedItem = None
        if acceptedItem is not None:
            ev.currentItem = acceptedItem
            try:
                acceptedItem.mouseClickEvent(ev)
            except:
                debug.printExc('Error sending click event:')
        else:
            for item in self.itemsNearEvent(ev):
                if not item.isVisible() or not item.isEnabled():
                    continue
                if hasattr(item, 'mouseClickEvent'):
                    ev.currentItem = item
                    try:
                        item.mouseClickEvent(ev)
                    except:
                        debug.printExc('Error sending click event:')
                    if ev.isAccepted():
                        if item.flags() & item.GraphicsItemFlag.ItemIsFocusable:
                            item.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                        break
    self.sigMouseClicked.emit(ev)
    return ev.isAccepted()