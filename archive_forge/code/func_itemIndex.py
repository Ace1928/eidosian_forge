from .. import functions as fn
from ..Qt import QtWidgets
from .GraphicsWidget import GraphicsWidget
from .LabelItem import LabelItem
from .PlotItem import PlotItem
from .ViewBox import ViewBox
def itemIndex(self, item):
    """Return the numerical index of GraphicsItem object passed in

        Parameters
        ----------
        item : QGraphicsLayoutItem
            Item to query the index position of

        Returns
        -------
        int
            Index of the item within the graphics layout

        Raises
        ------
        ValueError
            Raised if item could not be found inside the GraphicsLayout instance.
        """
    for i in range(self.layout.count()):
        if self.layout.itemAt(i).graphicsItem() is item:
            return i
    raise ValueError(f'Could not determine index of item {item}')