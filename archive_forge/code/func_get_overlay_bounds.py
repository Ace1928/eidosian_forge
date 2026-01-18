import param
from . import Dataset, util
from .dimension import ViewableElement
from .element import Element
from .layout import Layout
from .options import Store
from .overlay import NdOverlay, Overlay
from .spaces import Callable, HoloMap
@classmethod
def get_overlay_bounds(cls, overlay):
    """
        Returns the extents if all the elements of an overlay agree on
        a consistent extents, otherwise raises an exception.
        """
    if all((el.bounds == overlay.get(0).bounds for el in overlay)):
        return overlay.get(0).bounds
    else:
        raise ValueError('Extents across the overlay are inconsistent')