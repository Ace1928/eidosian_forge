import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def enable_goocanvas():
    if _check_enabled('goocanvas'):
        return
    gi.require_version('GooCanvas', '2.0')
    from gi.repository import GooCanvas
    _patch_module('goocanvas', GooCanvas)
    _install_enums(GooCanvas, strip='GOO_CANVAS_')
    _patch(GooCanvas, 'ItemSimple', GooCanvas.CanvasItemSimple)
    _patch(GooCanvas, 'Item', GooCanvas.CanvasItem)
    _patch(GooCanvas, 'Image', GooCanvas.CanvasImage)
    _patch(GooCanvas, 'Group', GooCanvas.CanvasGroup)
    _patch(GooCanvas, 'Rect', GooCanvas.CanvasRect)