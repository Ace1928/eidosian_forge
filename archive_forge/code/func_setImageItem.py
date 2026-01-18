import math
import weakref
import numpy as np
from .. import colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from .LinearRegionItem import LinearRegionItem
from .PlotItem import PlotItem
def setImageItem(self, img, insert_in=None):
    """
        Assigns an item or list of items to be represented and controlled.
        Supported "image items": class:`~pyqtgraph.ImageItem`, class:`~pyqtgraph.PColorMeshItem`

        Parameters
        ----------
        image: :class:`~pyqtgraph.ImageItem` or list of :class:`~pyqtgraph.ImageItem`
            Assigns one or more image items to this ColorBarItem.
            If a :class:`~pyqtgraph.ColorMap` is defined for ColorBarItem, this will be assigned to the 
            ImageItems. Otherwise, the ColorBarItem will attempt to retrieve a color map from the image items.
            In interactive mode, ColorBarItem will control the levels of the assigned image items, 
            simultaneously if there is more than one.
            If the ColorBarItem was initialized without a specified ``values`` parameter, it will attempt 
            to retrieve a set of user-defined ``levels`` from one of the image items. If this fails,
            the default values of ColorBarItem will be used as the (min, max) levels of the colorbar. 
            Note that, for non-interactive ColorBarItems, levels may be overridden by image items with 
            auto-scaling colors (defined by ``enableAutoLevels``). When using an interactive ColorBarItem
            in an animated plot, auto-scaling for its assigned image items should be *manually* disabled.
        insert_in: :class:`~pyqtgraph.PlotItem`, optional
            If a PlotItem is given, the color bar is inserted on the right
            or bottom of the plot, depending on the specified orientation.
        """
    try:
        self.img_list = [weakref.ref(item) for item in img]
    except TypeError:
        self.img_list = [weakref.ref(img)]
    colormap_is_undefined = self._colorMap is None
    for img_weakref in self.img_list:
        img = img_weakref()
        if img is not None:
            if hasattr(img, 'sigLevelsChanged'):
                img.sigLevelsChanged.connect(self._levelsChangedHandler)
            if colormap_is_undefined and hasattr(img, 'getColorMap'):
                img_cm = img.getColorMap()
                if img_cm is not None:
                    self._colorMap = img_cm
                    colormap_is_undefined = False
            if not self._actively_adjusted_values:
                if hasattr(img, 'getLevels'):
                    img_levels = img.getLevels()
                    if img_levels is not None:
                        self.setLevels(img_levels, update_items=False)
    if insert_in is not None:
        if self.horizontal:
            insert_in.layout.addItem(self, 5, 1)
            insert_in.layout.setRowFixedHeight(4, 10)
        else:
            insert_in.layout.addItem(self, 2, 5)
            insert_in.layout.setColumnFixedWidth(4, 5)
    self._update_items(update_cmap=True)