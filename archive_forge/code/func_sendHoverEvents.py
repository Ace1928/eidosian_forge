import warnings
import weakref
from time import perf_counter, perf_counter_ns
from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
def sendHoverEvents(self, ev, exitOnly=False):
    if exitOnly:
        acceptable = False
        items = []
        event = HoverEvent(None, acceptable)
    else:
        acceptable = not ev.buttons()
        event = HoverEvent(ev, acceptable)
        items = self.itemsNearEvent(event, hoverable=True)
        self.sigMouseHover.emit(items)
    prevItems = list(self.hoverItems.keys())
    for item in items:
        if hasattr(item, 'hoverEvent'):
            event.currentItem = item
            if item not in self.hoverItems:
                self.hoverItems[item] = None
                event.enter = True
            else:
                prevItems.remove(item)
                event.enter = False
            try:
                item.hoverEvent(event)
            except:
                debug.printExc('Error sending hover event:')
    event.enter = False
    event.exit = True
    for item in prevItems:
        event.currentItem = item
        try:
            if isQObjectAlive(item) and item.scene() is self:
                item.hoverEvent(event)
        except:
            debug.printExc('Error sending hover exit event:')
        finally:
            del self.hoverItems[item]
    if ev.type() == ev.Type.GraphicsSceneMousePress or (ev.type() == ev.Type.GraphicsSceneMouseMove and (not ev.buttons())):
        self.lastHoverEvent = event